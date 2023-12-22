from __future__ import annotations



def forward(self, primals_1: "f32[96]", primals_3: "f32[96, 1, 5, 5]", primals_4: "f32[54, 1, 7, 7]", primals_5: "f32[54, 1, 5, 5]", primals_6: "f32[54, 1, 3, 3]", primals_7: "f32[96, 1, 3, 3]", primals_8: "f32[54, 54, 1, 1]", primals_9: "f32[108, 1, 5, 5]", primals_10: "f32[108, 1, 7, 7]", primals_11: "f32[108, 1, 5, 5]", primals_12: "f32[108, 1, 3, 3]", primals_13: "f32[108, 1, 3, 3]", primals_14: "f32[108, 108, 1, 1]", primals_15: "f32[432, 1, 5, 5]", primals_16: "f32[432, 1, 7, 7]", primals_17: "f32[432, 1, 5, 5]", primals_18: "f32[432, 1, 3, 3]", primals_19: "f32[432, 1, 3, 3]", primals_20: "f32[432, 432, 1, 1]", primals_21: "f32[864, 1, 5, 5]", primals_22: "f32[864, 1, 7, 7]", primals_23: "f32[864, 1, 5, 5]", primals_24: "f32[864, 1, 3, 3]", primals_25: "f32[864, 1, 3, 3]", primals_26: "f32[864, 864, 1, 1]", primals_27: "f32[96, 3, 3, 3]", primals_28: "f32[54, 96, 1, 1]", primals_29: "f32[54]", primals_31: "f32[54, 96, 1, 1]", primals_32: "f32[54]", primals_34: "f32[54, 1, 5, 5]", primals_35: "f32[54, 54, 1, 1]", primals_36: "f32[54]", primals_38: "f32[54, 96, 1, 1]", primals_39: "f32[54]", primals_41: "f32[54, 54, 1, 1]", primals_42: "f32[54]", primals_44: "f32[54, 1, 7, 7]", primals_45: "f32[54, 54, 1, 1]", primals_46: "f32[54]", primals_48: "f32[54, 54, 1, 1]", primals_49: "f32[54]", primals_51: "f32[54, 1, 5, 5]", primals_52: "f32[54, 54, 1, 1]", primals_53: "f32[54]", primals_55: "f32[54, 54, 1, 1]", primals_56: "f32[54]", primals_58: "f32[54, 1, 3, 3]", primals_59: "f32[54, 54, 1, 1]", primals_60: "f32[54]", primals_62: "f32[54, 1, 3, 3]", primals_63: "f32[54, 54, 1, 1]", primals_64: "f32[54]", primals_66: "f32[54, 1, 3, 3]", primals_67: "f32[54, 54, 1, 1]", primals_68: "f32[54]", primals_70: "f32[54, 96, 1, 1]", primals_71: "f32[54]", primals_73: "f32[54, 1, 3, 3]", primals_74: "f32[54, 54, 1, 1]", primals_75: "f32[54]", primals_77: "f32[54]", primals_79: "f32[54, 96, 1, 1]", primals_80: "f32[54, 96, 1, 1]", primals_81: "f32[108]", primals_83: "f32[108, 270, 1, 1]", primals_84: "f32[108]", primals_86: "f32[108, 108, 1, 1]", primals_87: "f32[108]", primals_89: "f32[108, 1, 5, 5]", primals_90: "f32[108, 108, 1, 1]", primals_91: "f32[108]", primals_93: "f32[108, 108, 1, 1]", primals_94: "f32[108]", primals_96: "f32[108, 1, 7, 7]", primals_97: "f32[108, 108, 1, 1]", primals_98: "f32[108]", primals_100: "f32[108, 108, 1, 1]", primals_101: "f32[108]", primals_103: "f32[108, 1, 5, 5]", primals_104: "f32[108, 108, 1, 1]", primals_105: "f32[108]", primals_107: "f32[108, 108, 1, 1]", primals_108: "f32[108]", primals_110: "f32[108, 1, 3, 3]", primals_111: "f32[108, 108, 1, 1]", primals_112: "f32[108]", primals_114: "f32[108, 1, 3, 3]", primals_115: "f32[108, 108, 1, 1]", primals_116: "f32[108]", primals_118: "f32[108, 1, 3, 3]", primals_119: "f32[108, 108, 1, 1]", primals_120: "f32[108]", primals_122: "f32[108, 108, 1, 1]", primals_123: "f32[108]", primals_125: "f32[108, 1, 3, 3]", primals_126: "f32[108, 108, 1, 1]", primals_127: "f32[108]", primals_129: "f32[108]", primals_131: "f32[108, 270, 1, 1]", primals_132: "f32[108, 270, 1, 1]", primals_133: "f32[216]", primals_135: "f32[216, 540, 1, 1]", primals_136: "f32[216]", primals_138: "f32[216, 1, 5, 5]", primals_139: "f32[216, 216, 1, 1]", primals_140: "f32[216]", primals_142: "f32[216, 1, 5, 5]", primals_143: "f32[216, 216, 1, 1]", primals_144: "f32[216]", primals_146: "f32[216, 1, 7, 7]", primals_147: "f32[216, 216, 1, 1]", primals_148: "f32[216]", primals_150: "f32[216, 1, 7, 7]", primals_151: "f32[216, 216, 1, 1]", primals_152: "f32[216]", primals_154: "f32[216, 1, 5, 5]", primals_155: "f32[216, 216, 1, 1]", primals_156: "f32[216]", primals_158: "f32[216, 1, 5, 5]", primals_159: "f32[216, 216, 1, 1]", primals_160: "f32[216]", primals_162: "f32[216, 1, 3, 3]", primals_163: "f32[216, 216, 1, 1]", primals_164: "f32[216]", primals_166: "f32[216, 1, 3, 3]", primals_167: "f32[216, 216, 1, 1]", primals_168: "f32[216]", primals_170: "f32[216, 1, 3, 3]", primals_171: "f32[216, 216, 1, 1]", primals_172: "f32[216]", primals_174: "f32[216, 1, 3, 3]", primals_175: "f32[216, 216, 1, 1]", primals_176: "f32[216]", primals_178: "f32[216, 1, 3, 3]", primals_179: "f32[216, 216, 1, 1]", primals_180: "f32[216]", primals_182: "f32[216, 1, 3, 3]", primals_183: "f32[216, 216, 1, 1]", primals_184: "f32[216]", primals_186: "f32[216, 540, 1, 1]", primals_187: "f32[216]", primals_189: "f32[216, 1080, 1, 1]", primals_190: "f32[216]", primals_192: "f32[216, 1, 5, 5]", primals_193: "f32[216, 216, 1, 1]", primals_194: "f32[216]", primals_196: "f32[216, 1, 5, 5]", primals_197: "f32[216, 216, 1, 1]", primals_198: "f32[216]", primals_200: "f32[216, 1, 7, 7]", primals_201: "f32[216, 216, 1, 1]", primals_202: "f32[216]", primals_204: "f32[216, 1, 7, 7]", primals_205: "f32[216, 216, 1, 1]", primals_206: "f32[216]", primals_208: "f32[216, 1, 5, 5]", primals_209: "f32[216, 216, 1, 1]", primals_210: "f32[216]", primals_212: "f32[216, 1, 5, 5]", primals_213: "f32[216, 216, 1, 1]", primals_214: "f32[216]", primals_216: "f32[216, 1, 3, 3]", primals_217: "f32[216, 216, 1, 1]", primals_218: "f32[216]", primals_220: "f32[216, 1, 3, 3]", primals_221: "f32[216, 216, 1, 1]", primals_222: "f32[216]", primals_224: "f32[216, 1, 3, 3]", primals_225: "f32[216, 216, 1, 1]", primals_226: "f32[216]", primals_228: "f32[216, 1, 3, 3]", primals_229: "f32[216, 216, 1, 1]", primals_230: "f32[216]", primals_232: "f32[216, 1, 3, 3]", primals_233: "f32[216, 216, 1, 1]", primals_234: "f32[216]", primals_236: "f32[216, 1, 3, 3]", primals_237: "f32[216, 216, 1, 1]", primals_238: "f32[216]", primals_240: "f32[216, 1080, 1, 1]", primals_241: "f32[216]", primals_243: "f32[216, 1080, 1, 1]", primals_244: "f32[216]", primals_246: "f32[216, 1, 5, 5]", primals_247: "f32[216, 216, 1, 1]", primals_248: "f32[216]", primals_250: "f32[216, 1, 5, 5]", primals_251: "f32[216, 216, 1, 1]", primals_252: "f32[216]", primals_254: "f32[216, 1, 7, 7]", primals_255: "f32[216, 216, 1, 1]", primals_256: "f32[216]", primals_258: "f32[216, 1, 7, 7]", primals_259: "f32[216, 216, 1, 1]", primals_260: "f32[216]", primals_262: "f32[216, 1, 5, 5]", primals_263: "f32[216, 216, 1, 1]", primals_264: "f32[216]", primals_266: "f32[216, 1, 5, 5]", primals_267: "f32[216, 216, 1, 1]", primals_268: "f32[216]", primals_270: "f32[216, 1, 3, 3]", primals_271: "f32[216, 216, 1, 1]", primals_272: "f32[216]", primals_274: "f32[216, 1, 3, 3]", primals_275: "f32[216, 216, 1, 1]", primals_276: "f32[216]", primals_278: "f32[216, 1, 3, 3]", primals_279: "f32[216, 216, 1, 1]", primals_280: "f32[216]", primals_282: "f32[216, 1, 3, 3]", primals_283: "f32[216, 216, 1, 1]", primals_284: "f32[216]", primals_286: "f32[216, 1, 3, 3]", primals_287: "f32[216, 216, 1, 1]", primals_288: "f32[216]", primals_290: "f32[216, 1, 3, 3]", primals_291: "f32[216, 216, 1, 1]", primals_292: "f32[216]", primals_294: "f32[216, 1080, 1, 1]", primals_295: "f32[216]", primals_297: "f32[216, 1080, 1, 1]", primals_298: "f32[216]", primals_300: "f32[216, 1, 5, 5]", primals_301: "f32[216, 216, 1, 1]", primals_302: "f32[216]", primals_304: "f32[216, 1, 5, 5]", primals_305: "f32[216, 216, 1, 1]", primals_306: "f32[216]", primals_308: "f32[216, 1, 7, 7]", primals_309: "f32[216, 216, 1, 1]", primals_310: "f32[216]", primals_312: "f32[216, 1, 7, 7]", primals_313: "f32[216, 216, 1, 1]", primals_314: "f32[216]", primals_316: "f32[216, 1, 5, 5]", primals_317: "f32[216, 216, 1, 1]", primals_318: "f32[216]", primals_320: "f32[216, 1, 5, 5]", primals_321: "f32[216, 216, 1, 1]", primals_322: "f32[216]", primals_324: "f32[216, 1, 3, 3]", primals_325: "f32[216, 216, 1, 1]", primals_326: "f32[216]", primals_328: "f32[216, 1, 3, 3]", primals_329: "f32[216, 216, 1, 1]", primals_330: "f32[216]", primals_332: "f32[216, 1, 3, 3]", primals_333: "f32[216, 216, 1, 1]", primals_334: "f32[216]", primals_336: "f32[216, 1, 3, 3]", primals_337: "f32[216, 216, 1, 1]", primals_338: "f32[216]", primals_340: "f32[216, 1, 3, 3]", primals_341: "f32[216, 216, 1, 1]", primals_342: "f32[216]", primals_344: "f32[216, 1, 3, 3]", primals_345: "f32[216, 216, 1, 1]", primals_346: "f32[216]", primals_348: "f32[432, 1080, 1, 1]", primals_349: "f32[432]", primals_351: "f32[432, 1080, 1, 1]", primals_352: "f32[432]", primals_354: "f32[432, 432, 1, 1]", primals_355: "f32[432]", primals_357: "f32[432, 1, 5, 5]", primals_358: "f32[432, 432, 1, 1]", primals_359: "f32[432]", primals_361: "f32[432, 432, 1, 1]", primals_362: "f32[432]", primals_364: "f32[432, 1, 7, 7]", primals_365: "f32[432, 432, 1, 1]", primals_366: "f32[432]", primals_368: "f32[432, 432, 1, 1]", primals_369: "f32[432]", primals_371: "f32[432, 1, 5, 5]", primals_372: "f32[432, 432, 1, 1]", primals_373: "f32[432]", primals_375: "f32[432, 432, 1, 1]", primals_376: "f32[432]", primals_378: "f32[432, 1, 3, 3]", primals_379: "f32[432, 432, 1, 1]", primals_380: "f32[432]", primals_382: "f32[432, 1, 3, 3]", primals_383: "f32[432, 432, 1, 1]", primals_384: "f32[432]", primals_386: "f32[432, 1, 3, 3]", primals_387: "f32[432, 432, 1, 1]", primals_388: "f32[432]", primals_390: "f32[432, 432, 1, 1]", primals_391: "f32[432]", primals_393: "f32[432, 1, 3, 3]", primals_394: "f32[432, 432, 1, 1]", primals_395: "f32[432]", primals_397: "f32[432]", primals_399: "f32[216, 1080, 1, 1]", primals_400: "f32[216, 1080, 1, 1]", primals_401: "f32[432]", primals_403: "f32[432, 2160, 1, 1]", primals_404: "f32[432]", primals_406: "f32[432, 1, 5, 5]", primals_407: "f32[432, 432, 1, 1]", primals_408: "f32[432]", primals_410: "f32[432, 1, 5, 5]", primals_411: "f32[432, 432, 1, 1]", primals_412: "f32[432]", primals_414: "f32[432, 1, 7, 7]", primals_415: "f32[432, 432, 1, 1]", primals_416: "f32[432]", primals_418: "f32[432, 1, 7, 7]", primals_419: "f32[432, 432, 1, 1]", primals_420: "f32[432]", primals_422: "f32[432, 1, 5, 5]", primals_423: "f32[432, 432, 1, 1]", primals_424: "f32[432]", primals_426: "f32[432, 1, 5, 5]", primals_427: "f32[432, 432, 1, 1]", primals_428: "f32[432]", primals_430: "f32[432, 1, 3, 3]", primals_431: "f32[432, 432, 1, 1]", primals_432: "f32[432]", primals_434: "f32[432, 1, 3, 3]", primals_435: "f32[432, 432, 1, 1]", primals_436: "f32[432]", primals_438: "f32[432, 1, 3, 3]", primals_439: "f32[432, 432, 1, 1]", primals_440: "f32[432]", primals_442: "f32[432, 1, 3, 3]", primals_443: "f32[432, 432, 1, 1]", primals_444: "f32[432]", primals_446: "f32[432, 1, 3, 3]", primals_447: "f32[432, 432, 1, 1]", primals_448: "f32[432]", primals_450: "f32[432, 1, 3, 3]", primals_451: "f32[432, 432, 1, 1]", primals_452: "f32[432]", primals_454: "f32[432, 2160, 1, 1]", primals_455: "f32[432]", primals_457: "f32[432, 2160, 1, 1]", primals_458: "f32[432]", primals_460: "f32[432, 1, 5, 5]", primals_461: "f32[432, 432, 1, 1]", primals_462: "f32[432]", primals_464: "f32[432, 1, 5, 5]", primals_465: "f32[432, 432, 1, 1]", primals_466: "f32[432]", primals_468: "f32[432, 1, 7, 7]", primals_469: "f32[432, 432, 1, 1]", primals_470: "f32[432]", primals_472: "f32[432, 1, 7, 7]", primals_473: "f32[432, 432, 1, 1]", primals_474: "f32[432]", primals_476: "f32[432, 1, 5, 5]", primals_477: "f32[432, 432, 1, 1]", primals_478: "f32[432]", primals_480: "f32[432, 1, 5, 5]", primals_481: "f32[432, 432, 1, 1]", primals_482: "f32[432]", primals_484: "f32[432, 1, 3, 3]", primals_485: "f32[432, 432, 1, 1]", primals_486: "f32[432]", primals_488: "f32[432, 1, 3, 3]", primals_489: "f32[432, 432, 1, 1]", primals_490: "f32[432]", primals_492: "f32[432, 1, 3, 3]", primals_493: "f32[432, 432, 1, 1]", primals_494: "f32[432]", primals_496: "f32[432, 1, 3, 3]", primals_497: "f32[432, 432, 1, 1]", primals_498: "f32[432]", primals_500: "f32[432, 1, 3, 3]", primals_501: "f32[432, 432, 1, 1]", primals_502: "f32[432]", primals_504: "f32[432, 1, 3, 3]", primals_505: "f32[432, 432, 1, 1]", primals_506: "f32[432]", primals_508: "f32[432, 2160, 1, 1]", primals_509: "f32[432]", primals_511: "f32[432, 2160, 1, 1]", primals_512: "f32[432]", primals_514: "f32[432, 1, 5, 5]", primals_515: "f32[432, 432, 1, 1]", primals_516: "f32[432]", primals_518: "f32[432, 1, 5, 5]", primals_519: "f32[432, 432, 1, 1]", primals_520: "f32[432]", primals_522: "f32[432, 1, 7, 7]", primals_523: "f32[432, 432, 1, 1]", primals_524: "f32[432]", primals_526: "f32[432, 1, 7, 7]", primals_527: "f32[432, 432, 1, 1]", primals_528: "f32[432]", primals_530: "f32[432, 1, 5, 5]", primals_531: "f32[432, 432, 1, 1]", primals_532: "f32[432]", primals_534: "f32[432, 1, 5, 5]", primals_535: "f32[432, 432, 1, 1]", primals_536: "f32[432]", primals_538: "f32[432, 1, 3, 3]", primals_539: "f32[432, 432, 1, 1]", primals_540: "f32[432]", primals_542: "f32[432, 1, 3, 3]", primals_543: "f32[432, 432, 1, 1]", primals_544: "f32[432]", primals_546: "f32[432, 1, 3, 3]", primals_547: "f32[432, 432, 1, 1]", primals_548: "f32[432]", primals_550: "f32[432, 1, 3, 3]", primals_551: "f32[432, 432, 1, 1]", primals_552: "f32[432]", primals_554: "f32[432, 1, 3, 3]", primals_555: "f32[432, 432, 1, 1]", primals_556: "f32[432]", primals_558: "f32[432, 1, 3, 3]", primals_559: "f32[432, 432, 1, 1]", primals_560: "f32[432]", primals_562: "f32[864, 2160, 1, 1]", primals_563: "f32[864]", primals_565: "f32[864, 2160, 1, 1]", primals_566: "f32[864]", primals_568: "f32[864, 864, 1, 1]", primals_569: "f32[864]", primals_571: "f32[864, 1, 5, 5]", primals_572: "f32[864, 864, 1, 1]", primals_573: "f32[864]", primals_575: "f32[864, 864, 1, 1]", primals_576: "f32[864]", primals_578: "f32[864, 1, 7, 7]", primals_579: "f32[864, 864, 1, 1]", primals_580: "f32[864]", primals_582: "f32[864, 864, 1, 1]", primals_583: "f32[864]", primals_585: "f32[864, 1, 5, 5]", primals_586: "f32[864, 864, 1, 1]", primals_587: "f32[864]", primals_589: "f32[864, 864, 1, 1]", primals_590: "f32[864]", primals_592: "f32[864, 1, 3, 3]", primals_593: "f32[864, 864, 1, 1]", primals_594: "f32[864]", primals_596: "f32[864, 1, 3, 3]", primals_597: "f32[864, 864, 1, 1]", primals_598: "f32[864]", primals_600: "f32[864, 1, 3, 3]", primals_601: "f32[864, 864, 1, 1]", primals_602: "f32[864]", primals_604: "f32[864, 864, 1, 1]", primals_605: "f32[864]", primals_607: "f32[864, 1, 3, 3]", primals_608: "f32[864, 864, 1, 1]", primals_609: "f32[864]", primals_611: "f32[864]", primals_613: "f32[432, 2160, 1, 1]", primals_614: "f32[432, 2160, 1, 1]", primals_615: "f32[864]", primals_617: "f32[864, 4320, 1, 1]", primals_618: "f32[864]", primals_620: "f32[864, 1, 5, 5]", primals_621: "f32[864, 864, 1, 1]", primals_622: "f32[864]", primals_624: "f32[864, 1, 5, 5]", primals_625: "f32[864, 864, 1, 1]", primals_626: "f32[864]", primals_628: "f32[864, 1, 7, 7]", primals_629: "f32[864, 864, 1, 1]", primals_630: "f32[864]", primals_632: "f32[864, 1, 7, 7]", primals_633: "f32[864, 864, 1, 1]", primals_634: "f32[864]", primals_636: "f32[864, 1, 5, 5]", primals_637: "f32[864, 864, 1, 1]", primals_638: "f32[864]", primals_640: "f32[864, 1, 5, 5]", primals_641: "f32[864, 864, 1, 1]", primals_642: "f32[864]", primals_644: "f32[864, 1, 3, 3]", primals_645: "f32[864, 864, 1, 1]", primals_646: "f32[864]", primals_648: "f32[864, 1, 3, 3]", primals_649: "f32[864, 864, 1, 1]", primals_650: "f32[864]", primals_652: "f32[864, 1, 3, 3]", primals_653: "f32[864, 864, 1, 1]", primals_654: "f32[864]", primals_656: "f32[864, 1, 3, 3]", primals_657: "f32[864, 864, 1, 1]", primals_658: "f32[864]", primals_660: "f32[864, 1, 3, 3]", primals_661: "f32[864, 864, 1, 1]", primals_662: "f32[864]", primals_664: "f32[864, 1, 3, 3]", primals_665: "f32[864, 864, 1, 1]", primals_666: "f32[864]", primals_668: "f32[864, 4320, 1, 1]", primals_669: "f32[864]", primals_671: "f32[864, 4320, 1, 1]", primals_672: "f32[864]", primals_674: "f32[864, 1, 5, 5]", primals_675: "f32[864, 864, 1, 1]", primals_676: "f32[864]", primals_678: "f32[864, 1, 5, 5]", primals_679: "f32[864, 864, 1, 1]", primals_680: "f32[864]", primals_682: "f32[864, 1, 7, 7]", primals_683: "f32[864, 864, 1, 1]", primals_684: "f32[864]", primals_686: "f32[864, 1, 7, 7]", primals_687: "f32[864, 864, 1, 1]", primals_688: "f32[864]", primals_690: "f32[864, 1, 5, 5]", primals_691: "f32[864, 864, 1, 1]", primals_692: "f32[864]", primals_694: "f32[864, 1, 5, 5]", primals_695: "f32[864, 864, 1, 1]", primals_696: "f32[864]", primals_698: "f32[864, 1, 3, 3]", primals_699: "f32[864, 864, 1, 1]", primals_700: "f32[864]", primals_702: "f32[864, 1, 3, 3]", primals_703: "f32[864, 864, 1, 1]", primals_704: "f32[864]", primals_706: "f32[864, 1, 3, 3]", primals_707: "f32[864, 864, 1, 1]", primals_708: "f32[864]", primals_710: "f32[864, 1, 3, 3]", primals_711: "f32[864, 864, 1, 1]", primals_712: "f32[864]", primals_714: "f32[864, 1, 3, 3]", primals_715: "f32[864, 864, 1, 1]", primals_716: "f32[864]", primals_718: "f32[864, 1, 3, 3]", primals_719: "f32[864, 864, 1, 1]", primals_720: "f32[864]", primals_722: "f32[864, 4320, 1, 1]", primals_723: "f32[864]", primals_725: "f32[864, 4320, 1, 1]", primals_726: "f32[864]", primals_728: "f32[864, 1, 5, 5]", primals_729: "f32[864, 864, 1, 1]", primals_730: "f32[864]", primals_732: "f32[864, 1, 5, 5]", primals_733: "f32[864, 864, 1, 1]", primals_734: "f32[864]", primals_736: "f32[864, 1, 7, 7]", primals_737: "f32[864, 864, 1, 1]", primals_738: "f32[864]", primals_740: "f32[864, 1, 7, 7]", primals_741: "f32[864, 864, 1, 1]", primals_742: "f32[864]", primals_744: "f32[864, 1, 5, 5]", primals_745: "f32[864, 864, 1, 1]", primals_746: "f32[864]", primals_748: "f32[864, 1, 5, 5]", primals_749: "f32[864, 864, 1, 1]", primals_750: "f32[864]", primals_752: "f32[864, 1, 3, 3]", primals_753: "f32[864, 864, 1, 1]", primals_754: "f32[864]", primals_756: "f32[864, 1, 3, 3]", primals_757: "f32[864, 864, 1, 1]", primals_758: "f32[864]", primals_760: "f32[864, 1, 3, 3]", primals_761: "f32[864, 864, 1, 1]", primals_762: "f32[864]", primals_764: "f32[864, 1, 3, 3]", primals_765: "f32[864, 864, 1, 1]", primals_766: "f32[864]", primals_768: "f32[864, 1, 3, 3]", primals_769: "f32[864, 864, 1, 1]", primals_770: "f32[864]", primals_772: "f32[864, 1, 3, 3]", primals_773: "f32[864, 864, 1, 1]", primals_774: "f32[864]", primals_1381: "f32[8, 3, 331, 331]", convolution: "f32[8, 96, 165, 165]", squeeze_1: "f32[96]", relu: "f32[8, 96, 165, 165]", convolution_1: "f32[8, 54, 165, 165]", squeeze_4: "f32[54]", constant_pad_nd: "f32[8, 96, 169, 169]", convolution_2: "f32[8, 96, 83, 83]", convolution_3: "f32[8, 54, 83, 83]", squeeze_7: "f32[54]", relu_2: "f32[8, 54, 83, 83]", convolution_4: "f32[8, 54, 83, 83]", convolution_5: "f32[8, 54, 83, 83]", squeeze_10: "f32[54]", constant_pad_nd_1: "f32[8, 96, 167, 167]", getitem_8: "f32[8, 96, 83, 83]", getitem_9: "i64[8, 96, 83, 83]", convolution_6: "f32[8, 54, 83, 83]", squeeze_13: "f32[54]", constant_pad_nd_2: "f32[8, 54, 171, 171]", convolution_7: "f32[8, 54, 83, 83]", convolution_8: "f32[8, 54, 83, 83]", squeeze_16: "f32[54]", relu_4: "f32[8, 54, 83, 83]", convolution_9: "f32[8, 54, 83, 83]", convolution_10: "f32[8, 54, 83, 83]", squeeze_19: "f32[54]", constant_pad_nd_3: "f32[8, 54, 167, 167]", getitem_17: "i64[8, 54, 83, 83]", constant_pad_nd_4: "f32[8, 54, 169, 169]", convolution_11: "f32[8, 54, 83, 83]", convolution_12: "f32[8, 54, 83, 83]", squeeze_22: "f32[54]", relu_6: "f32[8, 54, 83, 83]", convolution_13: "f32[8, 54, 83, 83]", convolution_14: "f32[8, 54, 83, 83]", squeeze_25: "f32[54]", constant_pad_nd_5: "f32[8, 54, 167, 167]", convolution_15: "f32[8, 54, 83, 83]", convolution_16: "f32[8, 54, 83, 83]", squeeze_28: "f32[54]", relu_8: "f32[8, 54, 83, 83]", convolution_17: "f32[8, 54, 83, 83]", convolution_18: "f32[8, 54, 83, 83]", squeeze_31: "f32[54]", relu_9: "f32[8, 54, 83, 83]", convolution_19: "f32[8, 54, 83, 83]", convolution_20: "f32[8, 54, 83, 83]", squeeze_34: "f32[54]", relu_10: "f32[8, 54, 83, 83]", convolution_21: "f32[8, 54, 83, 83]", convolution_22: "f32[8, 54, 83, 83]", squeeze_37: "f32[54]", constant_pad_nd_7: "f32[8, 96, 167, 167]", convolution_23: "f32[8, 96, 83, 83]", convolution_24: "f32[8, 54, 83, 83]", squeeze_40: "f32[54]", relu_12: "f32[8, 54, 83, 83]", convolution_25: "f32[8, 54, 83, 83]", convolution_26: "f32[8, 54, 83, 83]", squeeze_43: "f32[54]", constant_pad_nd_8: "f32[8, 54, 165, 165]", convolution_27: "f32[8, 54, 83, 83]", squeeze_46: "f32[54]", avg_pool2d: "f32[8, 96, 83, 83]", constant_pad_nd_9: "f32[8, 96, 165, 165]", avg_pool2d_1: "f32[8, 96, 83, 83]", cat_1: "f32[8, 108, 83, 83]", squeeze_49: "f32[108]", relu_15: "f32[8, 270, 83, 83]", convolution_30: "f32[8, 108, 83, 83]", squeeze_52: "f32[108]", constant_pad_nd_10: "f32[8, 108, 87, 87]", convolution_31: "f32[8, 108, 42, 42]", convolution_32: "f32[8, 108, 42, 42]", squeeze_55: "f32[108]", relu_17: "f32[8, 108, 42, 42]", convolution_33: "f32[8, 108, 42, 42]", convolution_34: "f32[8, 108, 42, 42]", squeeze_58: "f32[108]", constant_pad_nd_11: "f32[8, 108, 85, 85]", getitem_47: "i64[8, 108, 42, 42]", constant_pad_nd_12: "f32[8, 108, 89, 89]", convolution_35: "f32[8, 108, 42, 42]", convolution_36: "f32[8, 108, 42, 42]", squeeze_61: "f32[108]", relu_19: "f32[8, 108, 42, 42]", convolution_37: "f32[8, 108, 42, 42]", convolution_38: "f32[8, 108, 42, 42]", squeeze_64: "f32[108]", constant_pad_nd_13: "f32[8, 108, 85, 85]", getitem_53: "i64[8, 108, 42, 42]", constant_pad_nd_14: "f32[8, 108, 87, 87]", convolution_39: "f32[8, 108, 42, 42]", convolution_40: "f32[8, 108, 42, 42]", squeeze_67: "f32[108]", relu_21: "f32[8, 108, 42, 42]", convolution_41: "f32[8, 108, 42, 42]", convolution_42: "f32[8, 108, 42, 42]", squeeze_70: "f32[108]", constant_pad_nd_15: "f32[8, 108, 85, 85]", convolution_43: "f32[8, 108, 42, 42]", convolution_44: "f32[8, 108, 42, 42]", squeeze_73: "f32[108]", relu_23: "f32[8, 108, 42, 42]", convolution_45: "f32[8, 108, 42, 42]", convolution_46: "f32[8, 108, 42, 42]", squeeze_76: "f32[108]", relu_24: "f32[8, 108, 42, 42]", convolution_47: "f32[8, 108, 42, 42]", convolution_48: "f32[8, 108, 42, 42]", squeeze_79: "f32[108]", relu_25: "f32[8, 108, 42, 42]", convolution_49: "f32[8, 108, 42, 42]", convolution_50: "f32[8, 108, 42, 42]", squeeze_82: "f32[108]", constant_pad_nd_17: "f32[8, 108, 85, 85]", convolution_51: "f32[8, 108, 42, 42]", convolution_52: "f32[8, 108, 42, 42]", squeeze_85: "f32[108]", relu_27: "f32[8, 108, 42, 42]", convolution_53: "f32[8, 108, 42, 42]", convolution_54: "f32[8, 108, 42, 42]", squeeze_88: "f32[108]", constant_pad_nd_18: "f32[8, 108, 83, 83]", convolution_55: "f32[8, 108, 42, 42]", squeeze_91: "f32[108]", avg_pool2d_2: "f32[8, 270, 42, 42]", constant_pad_nd_19: "f32[8, 270, 83, 83]", avg_pool2d_3: "f32[8, 270, 42, 42]", cat_3: "f32[8, 216, 42, 42]", squeeze_94: "f32[216]", add_169: "f32[8, 216, 42, 42]", relu_30: "f32[8, 540, 42, 42]", convolution_58: "f32[8, 216, 42, 42]", squeeze_97: "f32[216]", add_174: "f32[8, 216, 42, 42]", relu_31: "f32[8, 216, 42, 42]", convolution_59: "f32[8, 216, 42, 42]", convolution_60: "f32[8, 216, 42, 42]", squeeze_100: "f32[216]", relu_32: "f32[8, 216, 42, 42]", convolution_61: "f32[8, 216, 42, 42]", convolution_62: "f32[8, 216, 42, 42]", squeeze_103: "f32[216]", getitem_83: "i64[8, 216, 42, 42]", relu_33: "f32[8, 216, 42, 42]", convolution_63: "f32[8, 216, 42, 42]", convolution_64: "f32[8, 216, 42, 42]", squeeze_106: "f32[216]", relu_34: "f32[8, 216, 42, 42]", convolution_65: "f32[8, 216, 42, 42]", convolution_66: "f32[8, 216, 42, 42]", squeeze_109: "f32[216]", getitem_89: "i64[8, 216, 42, 42]", convolution_67: "f32[8, 216, 42, 42]", convolution_68: "f32[8, 216, 42, 42]", squeeze_112: "f32[216]", relu_36: "f32[8, 216, 42, 42]", convolution_69: "f32[8, 216, 42, 42]", convolution_70: "f32[8, 216, 42, 42]", squeeze_115: "f32[216]", convolution_71: "f32[8, 216, 42, 42]", convolution_72: "f32[8, 216, 42, 42]", squeeze_118: "f32[216]", relu_38: "f32[8, 216, 42, 42]", convolution_73: "f32[8, 216, 42, 42]", convolution_74: "f32[8, 216, 42, 42]", squeeze_121: "f32[216]", relu_39: "f32[8, 216, 42, 42]", convolution_75: "f32[8, 216, 42, 42]", convolution_76: "f32[8, 216, 42, 42]", squeeze_124: "f32[216]", relu_40: "f32[8, 216, 42, 42]", convolution_77: "f32[8, 216, 42, 42]", convolution_78: "f32[8, 216, 42, 42]", squeeze_127: "f32[216]", convolution_79: "f32[8, 216, 42, 42]", convolution_80: "f32[8, 216, 42, 42]", squeeze_130: "f32[216]", relu_42: "f32[8, 216, 42, 42]", convolution_81: "f32[8, 216, 42, 42]", convolution_82: "f32[8, 216, 42, 42]", squeeze_133: "f32[216]", convolution_83: "f32[8, 216, 42, 42]", squeeze_136: "f32[216]", add_244: "f32[8, 216, 42, 42]", relu_44: "f32[8, 1080, 42, 42]", convolution_84: "f32[8, 216, 42, 42]", squeeze_139: "f32[216]", add_249: "f32[8, 216, 42, 42]", relu_45: "f32[8, 216, 42, 42]", convolution_85: "f32[8, 216, 42, 42]", convolution_86: "f32[8, 216, 42, 42]", squeeze_142: "f32[216]", relu_46: "f32[8, 216, 42, 42]", convolution_87: "f32[8, 216, 42, 42]", convolution_88: "f32[8, 216, 42, 42]", squeeze_145: "f32[216]", getitem_117: "i64[8, 216, 42, 42]", relu_47: "f32[8, 216, 42, 42]", convolution_89: "f32[8, 216, 42, 42]", convolution_90: "f32[8, 216, 42, 42]", squeeze_148: "f32[216]", relu_48: "f32[8, 216, 42, 42]", convolution_91: "f32[8, 216, 42, 42]", convolution_92: "f32[8, 216, 42, 42]", squeeze_151: "f32[216]", getitem_123: "i64[8, 216, 42, 42]", convolution_93: "f32[8, 216, 42, 42]", convolution_94: "f32[8, 216, 42, 42]", squeeze_154: "f32[216]", relu_50: "f32[8, 216, 42, 42]", convolution_95: "f32[8, 216, 42, 42]", convolution_96: "f32[8, 216, 42, 42]", squeeze_157: "f32[216]", convolution_97: "f32[8, 216, 42, 42]", convolution_98: "f32[8, 216, 42, 42]", squeeze_160: "f32[216]", relu_52: "f32[8, 216, 42, 42]", convolution_99: "f32[8, 216, 42, 42]", convolution_100: "f32[8, 216, 42, 42]", squeeze_163: "f32[216]", relu_53: "f32[8, 216, 42, 42]", convolution_101: "f32[8, 216, 42, 42]", convolution_102: "f32[8, 216, 42, 42]", squeeze_166: "f32[216]", relu_54: "f32[8, 216, 42, 42]", convolution_103: "f32[8, 216, 42, 42]", convolution_104: "f32[8, 216, 42, 42]", squeeze_169: "f32[216]", convolution_105: "f32[8, 216, 42, 42]", convolution_106: "f32[8, 216, 42, 42]", squeeze_172: "f32[216]", relu_56: "f32[8, 216, 42, 42]", convolution_107: "f32[8, 216, 42, 42]", convolution_108: "f32[8, 216, 42, 42]", squeeze_175: "f32[216]", convolution_109: "f32[8, 216, 42, 42]", squeeze_178: "f32[216]", add_319: "f32[8, 216, 42, 42]", relu_58: "f32[8, 1080, 42, 42]", convolution_110: "f32[8, 216, 42, 42]", squeeze_181: "f32[216]", add_324: "f32[8, 216, 42, 42]", relu_59: "f32[8, 216, 42, 42]", convolution_111: "f32[8, 216, 42, 42]", convolution_112: "f32[8, 216, 42, 42]", squeeze_184: "f32[216]", relu_60: "f32[8, 216, 42, 42]", convolution_113: "f32[8, 216, 42, 42]", convolution_114: "f32[8, 216, 42, 42]", squeeze_187: "f32[216]", getitem_151: "i64[8, 216, 42, 42]", relu_61: "f32[8, 216, 42, 42]", convolution_115: "f32[8, 216, 42, 42]", convolution_116: "f32[8, 216, 42, 42]", squeeze_190: "f32[216]", relu_62: "f32[8, 216, 42, 42]", convolution_117: "f32[8, 216, 42, 42]", convolution_118: "f32[8, 216, 42, 42]", squeeze_193: "f32[216]", getitem_157: "i64[8, 216, 42, 42]", convolution_119: "f32[8, 216, 42, 42]", convolution_120: "f32[8, 216, 42, 42]", squeeze_196: "f32[216]", relu_64: "f32[8, 216, 42, 42]", convolution_121: "f32[8, 216, 42, 42]", convolution_122: "f32[8, 216, 42, 42]", squeeze_199: "f32[216]", convolution_123: "f32[8, 216, 42, 42]", convolution_124: "f32[8, 216, 42, 42]", squeeze_202: "f32[216]", relu_66: "f32[8, 216, 42, 42]", convolution_125: "f32[8, 216, 42, 42]", convolution_126: "f32[8, 216, 42, 42]", squeeze_205: "f32[216]", relu_67: "f32[8, 216, 42, 42]", convolution_127: "f32[8, 216, 42, 42]", convolution_128: "f32[8, 216, 42, 42]", squeeze_208: "f32[216]", relu_68: "f32[8, 216, 42, 42]", convolution_129: "f32[8, 216, 42, 42]", convolution_130: "f32[8, 216, 42, 42]", squeeze_211: "f32[216]", convolution_131: "f32[8, 216, 42, 42]", convolution_132: "f32[8, 216, 42, 42]", squeeze_214: "f32[216]", relu_70: "f32[8, 216, 42, 42]", convolution_133: "f32[8, 216, 42, 42]", convolution_134: "f32[8, 216, 42, 42]", squeeze_217: "f32[216]", convolution_135: "f32[8, 216, 42, 42]", squeeze_220: "f32[216]", add_394: "f32[8, 216, 42, 42]", relu_72: "f32[8, 1080, 42, 42]", convolution_136: "f32[8, 216, 42, 42]", squeeze_223: "f32[216]", add_399: "f32[8, 216, 42, 42]", relu_73: "f32[8, 216, 42, 42]", convolution_137: "f32[8, 216, 42, 42]", convolution_138: "f32[8, 216, 42, 42]", squeeze_226: "f32[216]", relu_74: "f32[8, 216, 42, 42]", convolution_139: "f32[8, 216, 42, 42]", convolution_140: "f32[8, 216, 42, 42]", squeeze_229: "f32[216]", getitem_185: "i64[8, 216, 42, 42]", relu_75: "f32[8, 216, 42, 42]", convolution_141: "f32[8, 216, 42, 42]", convolution_142: "f32[8, 216, 42, 42]", squeeze_232: "f32[216]", relu_76: "f32[8, 216, 42, 42]", convolution_143: "f32[8, 216, 42, 42]", convolution_144: "f32[8, 216, 42, 42]", squeeze_235: "f32[216]", getitem_191: "i64[8, 216, 42, 42]", convolution_145: "f32[8, 216, 42, 42]", convolution_146: "f32[8, 216, 42, 42]", squeeze_238: "f32[216]", relu_78: "f32[8, 216, 42, 42]", convolution_147: "f32[8, 216, 42, 42]", convolution_148: "f32[8, 216, 42, 42]", squeeze_241: "f32[216]", convolution_149: "f32[8, 216, 42, 42]", convolution_150: "f32[8, 216, 42, 42]", squeeze_244: "f32[216]", relu_80: "f32[8, 216, 42, 42]", convolution_151: "f32[8, 216, 42, 42]", convolution_152: "f32[8, 216, 42, 42]", squeeze_247: "f32[216]", relu_81: "f32[8, 216, 42, 42]", convolution_153: "f32[8, 216, 42, 42]", convolution_154: "f32[8, 216, 42, 42]", squeeze_250: "f32[216]", relu_82: "f32[8, 216, 42, 42]", convolution_155: "f32[8, 216, 42, 42]", convolution_156: "f32[8, 216, 42, 42]", squeeze_253: "f32[216]", convolution_157: "f32[8, 216, 42, 42]", convolution_158: "f32[8, 216, 42, 42]", squeeze_256: "f32[216]", relu_84: "f32[8, 216, 42, 42]", convolution_159: "f32[8, 216, 42, 42]", convolution_160: "f32[8, 216, 42, 42]", squeeze_259: "f32[216]", convolution_161: "f32[8, 432, 42, 42]", squeeze_262: "f32[432]", relu_86: "f32[8, 1080, 42, 42]", convolution_162: "f32[8, 432, 42, 42]", squeeze_265: "f32[432]", constant_pad_nd_20: "f32[8, 432, 45, 45]", convolution_163: "f32[8, 432, 21, 21]", convolution_164: "f32[8, 432, 21, 21]", squeeze_268: "f32[432]", relu_88: "f32[8, 432, 21, 21]", convolution_165: "f32[8, 432, 21, 21]", convolution_166: "f32[8, 432, 21, 21]", squeeze_271: "f32[432]", constant_pad_nd_21: "f32[8, 432, 43, 43]", getitem_219: "i64[8, 432, 21, 21]", constant_pad_nd_22: "f32[8, 432, 47, 47]", convolution_167: "f32[8, 432, 21, 21]", convolution_168: "f32[8, 432, 21, 21]", squeeze_274: "f32[432]", relu_90: "f32[8, 432, 21, 21]", convolution_169: "f32[8, 432, 21, 21]", convolution_170: "f32[8, 432, 21, 21]", squeeze_277: "f32[432]", constant_pad_nd_23: "f32[8, 432, 43, 43]", getitem_225: "i64[8, 432, 21, 21]", constant_pad_nd_24: "f32[8, 432, 45, 45]", convolution_171: "f32[8, 432, 21, 21]", convolution_172: "f32[8, 432, 21, 21]", squeeze_280: "f32[432]", relu_92: "f32[8, 432, 21, 21]", convolution_173: "f32[8, 432, 21, 21]", convolution_174: "f32[8, 432, 21, 21]", squeeze_283: "f32[432]", constant_pad_nd_25: "f32[8, 432, 43, 43]", convolution_175: "f32[8, 432, 21, 21]", convolution_176: "f32[8, 432, 21, 21]", squeeze_286: "f32[432]", relu_94: "f32[8, 432, 21, 21]", convolution_177: "f32[8, 432, 21, 21]", convolution_178: "f32[8, 432, 21, 21]", squeeze_289: "f32[432]", relu_95: "f32[8, 432, 21, 21]", convolution_179: "f32[8, 432, 21, 21]", convolution_180: "f32[8, 432, 21, 21]", squeeze_292: "f32[432]", relu_96: "f32[8, 432, 21, 21]", convolution_181: "f32[8, 432, 21, 21]", convolution_182: "f32[8, 432, 21, 21]", squeeze_295: "f32[432]", constant_pad_nd_27: "f32[8, 432, 43, 43]", convolution_183: "f32[8, 432, 21, 21]", convolution_184: "f32[8, 432, 21, 21]", squeeze_298: "f32[432]", relu_98: "f32[8, 432, 21, 21]", convolution_185: "f32[8, 432, 21, 21]", convolution_186: "f32[8, 432, 21, 21]", squeeze_301: "f32[432]", constant_pad_nd_28: "f32[8, 432, 42, 42]", convolution_187: "f32[8, 432, 21, 21]", squeeze_304: "f32[432]", avg_pool2d_4: "f32[8, 1080, 21, 21]", constant_pad_nd_29: "f32[8, 1080, 42, 42]", avg_pool2d_5: "f32[8, 1080, 21, 21]", cat_9: "f32[8, 432, 21, 21]", squeeze_307: "f32[432]", add_549: "f32[8, 432, 21, 21]", relu_101: "f32[8, 2160, 21, 21]", convolution_190: "f32[8, 432, 21, 21]", squeeze_310: "f32[432]", add_554: "f32[8, 432, 21, 21]", relu_102: "f32[8, 432, 21, 21]", convolution_191: "f32[8, 432, 21, 21]", convolution_192: "f32[8, 432, 21, 21]", squeeze_313: "f32[432]", relu_103: "f32[8, 432, 21, 21]", convolution_193: "f32[8, 432, 21, 21]", convolution_194: "f32[8, 432, 21, 21]", squeeze_316: "f32[432]", getitem_255: "i64[8, 432, 21, 21]", relu_104: "f32[8, 432, 21, 21]", convolution_195: "f32[8, 432, 21, 21]", convolution_196: "f32[8, 432, 21, 21]", squeeze_319: "f32[432]", relu_105: "f32[8, 432, 21, 21]", convolution_197: "f32[8, 432, 21, 21]", convolution_198: "f32[8, 432, 21, 21]", squeeze_322: "f32[432]", getitem_261: "i64[8, 432, 21, 21]", convolution_199: "f32[8, 432, 21, 21]", convolution_200: "f32[8, 432, 21, 21]", squeeze_325: "f32[432]", relu_107: "f32[8, 432, 21, 21]", convolution_201: "f32[8, 432, 21, 21]", convolution_202: "f32[8, 432, 21, 21]", squeeze_328: "f32[432]", convolution_203: "f32[8, 432, 21, 21]", convolution_204: "f32[8, 432, 21, 21]", squeeze_331: "f32[432]", relu_109: "f32[8, 432, 21, 21]", convolution_205: "f32[8, 432, 21, 21]", convolution_206: "f32[8, 432, 21, 21]", squeeze_334: "f32[432]", relu_110: "f32[8, 432, 21, 21]", convolution_207: "f32[8, 432, 21, 21]", convolution_208: "f32[8, 432, 21, 21]", squeeze_337: "f32[432]", relu_111: "f32[8, 432, 21, 21]", convolution_209: "f32[8, 432, 21, 21]", convolution_210: "f32[8, 432, 21, 21]", squeeze_340: "f32[432]", convolution_211: "f32[8, 432, 21, 21]", convolution_212: "f32[8, 432, 21, 21]", squeeze_343: "f32[432]", relu_113: "f32[8, 432, 21, 21]", convolution_213: "f32[8, 432, 21, 21]", convolution_214: "f32[8, 432, 21, 21]", squeeze_346: "f32[432]", convolution_215: "f32[8, 432, 21, 21]", squeeze_349: "f32[432]", add_624: "f32[8, 432, 21, 21]", relu_115: "f32[8, 2160, 21, 21]", convolution_216: "f32[8, 432, 21, 21]", squeeze_352: "f32[432]", add_629: "f32[8, 432, 21, 21]", relu_116: "f32[8, 432, 21, 21]", convolution_217: "f32[8, 432, 21, 21]", convolution_218: "f32[8, 432, 21, 21]", squeeze_355: "f32[432]", relu_117: "f32[8, 432, 21, 21]", convolution_219: "f32[8, 432, 21, 21]", convolution_220: "f32[8, 432, 21, 21]", squeeze_358: "f32[432]", getitem_289: "i64[8, 432, 21, 21]", relu_118: "f32[8, 432, 21, 21]", convolution_221: "f32[8, 432, 21, 21]", convolution_222: "f32[8, 432, 21, 21]", squeeze_361: "f32[432]", relu_119: "f32[8, 432, 21, 21]", convolution_223: "f32[8, 432, 21, 21]", convolution_224: "f32[8, 432, 21, 21]", squeeze_364: "f32[432]", getitem_295: "i64[8, 432, 21, 21]", convolution_225: "f32[8, 432, 21, 21]", convolution_226: "f32[8, 432, 21, 21]", squeeze_367: "f32[432]", relu_121: "f32[8, 432, 21, 21]", convolution_227: "f32[8, 432, 21, 21]", convolution_228: "f32[8, 432, 21, 21]", squeeze_370: "f32[432]", convolution_229: "f32[8, 432, 21, 21]", convolution_230: "f32[8, 432, 21, 21]", squeeze_373: "f32[432]", relu_123: "f32[8, 432, 21, 21]", convolution_231: "f32[8, 432, 21, 21]", convolution_232: "f32[8, 432, 21, 21]", squeeze_376: "f32[432]", relu_124: "f32[8, 432, 21, 21]", convolution_233: "f32[8, 432, 21, 21]", convolution_234: "f32[8, 432, 21, 21]", squeeze_379: "f32[432]", relu_125: "f32[8, 432, 21, 21]", convolution_235: "f32[8, 432, 21, 21]", convolution_236: "f32[8, 432, 21, 21]", squeeze_382: "f32[432]", convolution_237: "f32[8, 432, 21, 21]", convolution_238: "f32[8, 432, 21, 21]", squeeze_385: "f32[432]", relu_127: "f32[8, 432, 21, 21]", convolution_239: "f32[8, 432, 21, 21]", convolution_240: "f32[8, 432, 21, 21]", squeeze_388: "f32[432]", convolution_241: "f32[8, 432, 21, 21]", squeeze_391: "f32[432]", add_699: "f32[8, 432, 21, 21]", relu_129: "f32[8, 2160, 21, 21]", convolution_242: "f32[8, 432, 21, 21]", squeeze_394: "f32[432]", add_704: "f32[8, 432, 21, 21]", relu_130: "f32[8, 432, 21, 21]", convolution_243: "f32[8, 432, 21, 21]", convolution_244: "f32[8, 432, 21, 21]", squeeze_397: "f32[432]", relu_131: "f32[8, 432, 21, 21]", convolution_245: "f32[8, 432, 21, 21]", convolution_246: "f32[8, 432, 21, 21]", squeeze_400: "f32[432]", getitem_323: "i64[8, 432, 21, 21]", relu_132: "f32[8, 432, 21, 21]", convolution_247: "f32[8, 432, 21, 21]", convolution_248: "f32[8, 432, 21, 21]", squeeze_403: "f32[432]", relu_133: "f32[8, 432, 21, 21]", convolution_249: "f32[8, 432, 21, 21]", convolution_250: "f32[8, 432, 21, 21]", squeeze_406: "f32[432]", getitem_329: "i64[8, 432, 21, 21]", convolution_251: "f32[8, 432, 21, 21]", convolution_252: "f32[8, 432, 21, 21]", squeeze_409: "f32[432]", relu_135: "f32[8, 432, 21, 21]", convolution_253: "f32[8, 432, 21, 21]", convolution_254: "f32[8, 432, 21, 21]", squeeze_412: "f32[432]", convolution_255: "f32[8, 432, 21, 21]", convolution_256: "f32[8, 432, 21, 21]", squeeze_415: "f32[432]", relu_137: "f32[8, 432, 21, 21]", convolution_257: "f32[8, 432, 21, 21]", convolution_258: "f32[8, 432, 21, 21]", squeeze_418: "f32[432]", relu_138: "f32[8, 432, 21, 21]", convolution_259: "f32[8, 432, 21, 21]", convolution_260: "f32[8, 432, 21, 21]", squeeze_421: "f32[432]", relu_139: "f32[8, 432, 21, 21]", convolution_261: "f32[8, 432, 21, 21]", convolution_262: "f32[8, 432, 21, 21]", squeeze_424: "f32[432]", convolution_263: "f32[8, 432, 21, 21]", convolution_264: "f32[8, 432, 21, 21]", squeeze_427: "f32[432]", relu_141: "f32[8, 432, 21, 21]", convolution_265: "f32[8, 432, 21, 21]", convolution_266: "f32[8, 432, 21, 21]", squeeze_430: "f32[432]", convolution_267: "f32[8, 864, 21, 21]", squeeze_433: "f32[864]", relu_143: "f32[8, 2160, 21, 21]", convolution_268: "f32[8, 864, 21, 21]", squeeze_436: "f32[864]", constant_pad_nd_30: "f32[8, 864, 25, 25]", convolution_269: "f32[8, 864, 11, 11]", convolution_270: "f32[8, 864, 11, 11]", squeeze_439: "f32[864]", relu_145: "f32[8, 864, 11, 11]", convolution_271: "f32[8, 864, 11, 11]", convolution_272: "f32[8, 864, 11, 11]", squeeze_442: "f32[864]", constant_pad_nd_31: "f32[8, 864, 23, 23]", getitem_357: "i64[8, 864, 11, 11]", constant_pad_nd_32: "f32[8, 864, 27, 27]", convolution_273: "f32[8, 864, 11, 11]", convolution_274: "f32[8, 864, 11, 11]", squeeze_445: "f32[864]", relu_147: "f32[8, 864, 11, 11]", convolution_275: "f32[8, 864, 11, 11]", convolution_276: "f32[8, 864, 11, 11]", squeeze_448: "f32[864]", constant_pad_nd_33: "f32[8, 864, 23, 23]", getitem_363: "i64[8, 864, 11, 11]", constant_pad_nd_34: "f32[8, 864, 25, 25]", convolution_277: "f32[8, 864, 11, 11]", convolution_278: "f32[8, 864, 11, 11]", squeeze_451: "f32[864]", relu_149: "f32[8, 864, 11, 11]", convolution_279: "f32[8, 864, 11, 11]", convolution_280: "f32[8, 864, 11, 11]", squeeze_454: "f32[864]", constant_pad_nd_35: "f32[8, 864, 23, 23]", convolution_281: "f32[8, 864, 11, 11]", convolution_282: "f32[8, 864, 11, 11]", squeeze_457: "f32[864]", relu_151: "f32[8, 864, 11, 11]", convolution_283: "f32[8, 864, 11, 11]", convolution_284: "f32[8, 864, 11, 11]", squeeze_460: "f32[864]", relu_152: "f32[8, 864, 11, 11]", convolution_285: "f32[8, 864, 11, 11]", convolution_286: "f32[8, 864, 11, 11]", squeeze_463: "f32[864]", relu_153: "f32[8, 864, 11, 11]", convolution_287: "f32[8, 864, 11, 11]", convolution_288: "f32[8, 864, 11, 11]", squeeze_466: "f32[864]", constant_pad_nd_37: "f32[8, 864, 23, 23]", convolution_289: "f32[8, 864, 11, 11]", convolution_290: "f32[8, 864, 11, 11]", squeeze_469: "f32[864]", relu_155: "f32[8, 864, 11, 11]", convolution_291: "f32[8, 864, 11, 11]", convolution_292: "f32[8, 864, 11, 11]", squeeze_472: "f32[864]", constant_pad_nd_38: "f32[8, 864, 21, 21]", convolution_293: "f32[8, 864, 11, 11]", squeeze_475: "f32[864]", avg_pool2d_6: "f32[8, 2160, 11, 11]", constant_pad_nd_39: "f32[8, 2160, 21, 21]", avg_pool2d_7: "f32[8, 2160, 11, 11]", cat_14: "f32[8, 864, 11, 11]", squeeze_478: "f32[864]", add_854: "f32[8, 864, 11, 11]", relu_158: "f32[8, 4320, 11, 11]", convolution_296: "f32[8, 864, 11, 11]", squeeze_481: "f32[864]", add_859: "f32[8, 864, 11, 11]", relu_159: "f32[8, 864, 11, 11]", convolution_297: "f32[8, 864, 11, 11]", convolution_298: "f32[8, 864, 11, 11]", squeeze_484: "f32[864]", relu_160: "f32[8, 864, 11, 11]", convolution_299: "f32[8, 864, 11, 11]", convolution_300: "f32[8, 864, 11, 11]", squeeze_487: "f32[864]", getitem_393: "i64[8, 864, 11, 11]", relu_161: "f32[8, 864, 11, 11]", convolution_301: "f32[8, 864, 11, 11]", convolution_302: "f32[8, 864, 11, 11]", squeeze_490: "f32[864]", relu_162: "f32[8, 864, 11, 11]", convolution_303: "f32[8, 864, 11, 11]", convolution_304: "f32[8, 864, 11, 11]", squeeze_493: "f32[864]", getitem_399: "i64[8, 864, 11, 11]", convolution_305: "f32[8, 864, 11, 11]", convolution_306: "f32[8, 864, 11, 11]", squeeze_496: "f32[864]", relu_164: "f32[8, 864, 11, 11]", convolution_307: "f32[8, 864, 11, 11]", convolution_308: "f32[8, 864, 11, 11]", squeeze_499: "f32[864]", convolution_309: "f32[8, 864, 11, 11]", convolution_310: "f32[8, 864, 11, 11]", squeeze_502: "f32[864]", relu_166: "f32[8, 864, 11, 11]", convolution_311: "f32[8, 864, 11, 11]", convolution_312: "f32[8, 864, 11, 11]", squeeze_505: "f32[864]", relu_167: "f32[8, 864, 11, 11]", convolution_313: "f32[8, 864, 11, 11]", convolution_314: "f32[8, 864, 11, 11]", squeeze_508: "f32[864]", relu_168: "f32[8, 864, 11, 11]", convolution_315: "f32[8, 864, 11, 11]", convolution_316: "f32[8, 864, 11, 11]", squeeze_511: "f32[864]", convolution_317: "f32[8, 864, 11, 11]", convolution_318: "f32[8, 864, 11, 11]", squeeze_514: "f32[864]", relu_170: "f32[8, 864, 11, 11]", convolution_319: "f32[8, 864, 11, 11]", convolution_320: "f32[8, 864, 11, 11]", squeeze_517: "f32[864]", convolution_321: "f32[8, 864, 11, 11]", squeeze_520: "f32[864]", add_929: "f32[8, 864, 11, 11]", relu_172: "f32[8, 4320, 11, 11]", convolution_322: "f32[8, 864, 11, 11]", squeeze_523: "f32[864]", add_934: "f32[8, 864, 11, 11]", relu_173: "f32[8, 864, 11, 11]", convolution_323: "f32[8, 864, 11, 11]", convolution_324: "f32[8, 864, 11, 11]", squeeze_526: "f32[864]", relu_174: "f32[8, 864, 11, 11]", convolution_325: "f32[8, 864, 11, 11]", convolution_326: "f32[8, 864, 11, 11]", squeeze_529: "f32[864]", getitem_427: "i64[8, 864, 11, 11]", relu_175: "f32[8, 864, 11, 11]", convolution_327: "f32[8, 864, 11, 11]", convolution_328: "f32[8, 864, 11, 11]", squeeze_532: "f32[864]", relu_176: "f32[8, 864, 11, 11]", convolution_329: "f32[8, 864, 11, 11]", convolution_330: "f32[8, 864, 11, 11]", squeeze_535: "f32[864]", getitem_433: "i64[8, 864, 11, 11]", convolution_331: "f32[8, 864, 11, 11]", convolution_332: "f32[8, 864, 11, 11]", squeeze_538: "f32[864]", relu_178: "f32[8, 864, 11, 11]", convolution_333: "f32[8, 864, 11, 11]", convolution_334: "f32[8, 864, 11, 11]", squeeze_541: "f32[864]", convolution_335: "f32[8, 864, 11, 11]", convolution_336: "f32[8, 864, 11, 11]", squeeze_544: "f32[864]", relu_180: "f32[8, 864, 11, 11]", convolution_337: "f32[8, 864, 11, 11]", convolution_338: "f32[8, 864, 11, 11]", squeeze_547: "f32[864]", relu_181: "f32[8, 864, 11, 11]", convolution_339: "f32[8, 864, 11, 11]", convolution_340: "f32[8, 864, 11, 11]", squeeze_550: "f32[864]", relu_182: "f32[8, 864, 11, 11]", convolution_341: "f32[8, 864, 11, 11]", convolution_342: "f32[8, 864, 11, 11]", squeeze_553: "f32[864]", convolution_343: "f32[8, 864, 11, 11]", convolution_344: "f32[8, 864, 11, 11]", squeeze_556: "f32[864]", relu_184: "f32[8, 864, 11, 11]", convolution_345: "f32[8, 864, 11, 11]", convolution_346: "f32[8, 864, 11, 11]", squeeze_559: "f32[864]", convolution_347: "f32[8, 864, 11, 11]", squeeze_562: "f32[864]", add_1004: "f32[8, 864, 11, 11]", relu_186: "f32[8, 4320, 11, 11]", convolution_348: "f32[8, 864, 11, 11]", squeeze_565: "f32[864]", add_1009: "f32[8, 864, 11, 11]", relu_187: "f32[8, 864, 11, 11]", convolution_349: "f32[8, 864, 11, 11]", convolution_350: "f32[8, 864, 11, 11]", squeeze_568: "f32[864]", relu_188: "f32[8, 864, 11, 11]", convolution_351: "f32[8, 864, 11, 11]", convolution_352: "f32[8, 864, 11, 11]", squeeze_571: "f32[864]", getitem_461: "i64[8, 864, 11, 11]", relu_189: "f32[8, 864, 11, 11]", convolution_353: "f32[8, 864, 11, 11]", convolution_354: "f32[8, 864, 11, 11]", squeeze_574: "f32[864]", relu_190: "f32[8, 864, 11, 11]", convolution_355: "f32[8, 864, 11, 11]", convolution_356: "f32[8, 864, 11, 11]", squeeze_577: "f32[864]", getitem_467: "i64[8, 864, 11, 11]", convolution_357: "f32[8, 864, 11, 11]", convolution_358: "f32[8, 864, 11, 11]", squeeze_580: "f32[864]", relu_192: "f32[8, 864, 11, 11]", convolution_359: "f32[8, 864, 11, 11]", convolution_360: "f32[8, 864, 11, 11]", squeeze_583: "f32[864]", convolution_361: "f32[8, 864, 11, 11]", convolution_362: "f32[8, 864, 11, 11]", squeeze_586: "f32[864]", relu_194: "f32[8, 864, 11, 11]", convolution_363: "f32[8, 864, 11, 11]", convolution_364: "f32[8, 864, 11, 11]", squeeze_589: "f32[864]", relu_195: "f32[8, 864, 11, 11]", convolution_365: "f32[8, 864, 11, 11]", convolution_366: "f32[8, 864, 11, 11]", squeeze_592: "f32[864]", relu_196: "f32[8, 864, 11, 11]", convolution_367: "f32[8, 864, 11, 11]", convolution_368: "f32[8, 864, 11, 11]", squeeze_595: "f32[864]", convolution_369: "f32[8, 864, 11, 11]", convolution_370: "f32[8, 864, 11, 11]", squeeze_598: "f32[864]", relu_198: "f32[8, 864, 11, 11]", convolution_371: "f32[8, 864, 11, 11]", convolution_372: "f32[8, 864, 11, 11]", squeeze_601: "f32[864]", clone: "f32[8, 4320]", permute_1: "f32[1000, 4320]", le: "b8[8, 4320, 11, 11]", unsqueeze_806: "f32[1, 864, 1, 1]", unsqueeze_818: "f32[1, 864, 1, 1]", unsqueeze_830: "f32[1, 864, 1, 1]", unsqueeze_842: "f32[1, 864, 1, 1]", unsqueeze_854: "f32[1, 864, 1, 1]", unsqueeze_866: "f32[1, 864, 1, 1]", unsqueeze_878: "f32[1, 864, 1, 1]", unsqueeze_890: "f32[1, 864, 1, 1]", unsqueeze_902: "f32[1, 864, 1, 1]", unsqueeze_914: "f32[1, 864, 1, 1]", unsqueeze_926: "f32[1, 864, 1, 1]", unsqueeze_938: "f32[1, 864, 1, 1]", unsqueeze_950: "f32[1, 864, 1, 1]", unsqueeze_962: "f32[1, 864, 1, 1]", unsqueeze_974: "f32[1, 864, 1, 1]", unsqueeze_986: "f32[1, 864, 1, 1]", unsqueeze_998: "f32[1, 864, 1, 1]", unsqueeze_1010: "f32[1, 864, 1, 1]", unsqueeze_1022: "f32[1, 864, 1, 1]", unsqueeze_1034: "f32[1, 864, 1, 1]", unsqueeze_1046: "f32[1, 864, 1, 1]", unsqueeze_1058: "f32[1, 864, 1, 1]", unsqueeze_1070: "f32[1, 864, 1, 1]", unsqueeze_1082: "f32[1, 864, 1, 1]", unsqueeze_1094: "f32[1, 864, 1, 1]", unsqueeze_1106: "f32[1, 864, 1, 1]", unsqueeze_1118: "f32[1, 864, 1, 1]", unsqueeze_1130: "f32[1, 864, 1, 1]", unsqueeze_1142: "f32[1, 864, 1, 1]", unsqueeze_1154: "f32[1, 864, 1, 1]", unsqueeze_1166: "f32[1, 864, 1, 1]", unsqueeze_1178: "f32[1, 864, 1, 1]", unsqueeze_1190: "f32[1, 864, 1, 1]", unsqueeze_1202: "f32[1, 864, 1, 1]", unsqueeze_1214: "f32[1, 864, 1, 1]", unsqueeze_1226: "f32[1, 864, 1, 1]", unsqueeze_1238: "f32[1, 864, 1, 1]", unsqueeze_1250: "f32[1, 864, 1, 1]", unsqueeze_1262: "f32[1, 864, 1, 1]", unsqueeze_1274: "f32[1, 864, 1, 1]", unsqueeze_1286: "f32[1, 864, 1, 1]", unsqueeze_1298: "f32[1, 864, 1, 1]", unsqueeze_1310: "f32[1, 864, 1, 1]", le_43: "b8[8, 864, 21, 21]", unsqueeze_1322: "f32[1, 864, 1, 1]", unsqueeze_1334: "f32[1, 864, 1, 1]", le_45: "b8[8, 864, 21, 21]", unsqueeze_1346: "f32[1, 864, 1, 1]", unsqueeze_1358: "f32[1, 864, 1, 1]", unsqueeze_1370: "f32[1, 864, 1, 1]", unsqueeze_1382: "f32[1, 864, 1, 1]", unsqueeze_1394: "f32[1, 864, 1, 1]", unsqueeze_1406: "f32[1, 864, 1, 1]", unsqueeze_1418: "f32[1, 864, 1, 1]", unsqueeze_1430: "f32[1, 864, 1, 1]", unsqueeze_1442: "f32[1, 864, 1, 1]", unsqueeze_1454: "f32[1, 864, 1, 1]", unsqueeze_1466: "f32[1, 864, 1, 1]", unsqueeze_1478: "f32[1, 864, 1, 1]", unsqueeze_1490: "f32[1, 432, 1, 1]", unsqueeze_1502: "f32[1, 432, 1, 1]", unsqueeze_1514: "f32[1, 432, 1, 1]", unsqueeze_1526: "f32[1, 432, 1, 1]", unsqueeze_1538: "f32[1, 432, 1, 1]", unsqueeze_1550: "f32[1, 432, 1, 1]", unsqueeze_1562: "f32[1, 432, 1, 1]", unsqueeze_1574: "f32[1, 432, 1, 1]", unsqueeze_1586: "f32[1, 432, 1, 1]", unsqueeze_1598: "f32[1, 432, 1, 1]", unsqueeze_1610: "f32[1, 432, 1, 1]", unsqueeze_1622: "f32[1, 432, 1, 1]", unsqueeze_1634: "f32[1, 432, 1, 1]", unsqueeze_1646: "f32[1, 432, 1, 1]", unsqueeze_1658: "f32[1, 432, 1, 1]", unsqueeze_1670: "f32[1, 432, 1, 1]", unsqueeze_1682: "f32[1, 432, 1, 1]", unsqueeze_1694: "f32[1, 432, 1, 1]", unsqueeze_1706: "f32[1, 432, 1, 1]", unsqueeze_1718: "f32[1, 432, 1, 1]", unsqueeze_1730: "f32[1, 432, 1, 1]", unsqueeze_1742: "f32[1, 432, 1, 1]", unsqueeze_1754: "f32[1, 432, 1, 1]", unsqueeze_1766: "f32[1, 432, 1, 1]", unsqueeze_1778: "f32[1, 432, 1, 1]", unsqueeze_1790: "f32[1, 432, 1, 1]", unsqueeze_1802: "f32[1, 432, 1, 1]", unsqueeze_1814: "f32[1, 432, 1, 1]", unsqueeze_1826: "f32[1, 432, 1, 1]", unsqueeze_1838: "f32[1, 432, 1, 1]", unsqueeze_1850: "f32[1, 432, 1, 1]", unsqueeze_1862: "f32[1, 432, 1, 1]", unsqueeze_1874: "f32[1, 432, 1, 1]", unsqueeze_1886: "f32[1, 432, 1, 1]", unsqueeze_1898: "f32[1, 432, 1, 1]", unsqueeze_1910: "f32[1, 432, 1, 1]", unsqueeze_1922: "f32[1, 432, 1, 1]", unsqueeze_1934: "f32[1, 432, 1, 1]", unsqueeze_1946: "f32[1, 432, 1, 1]", unsqueeze_1958: "f32[1, 432, 1, 1]", unsqueeze_1970: "f32[1, 432, 1, 1]", unsqueeze_1982: "f32[1, 432, 1, 1]", unsqueeze_1994: "f32[1, 432, 1, 1]", le_100: "b8[8, 432, 42, 42]", unsqueeze_2006: "f32[1, 432, 1, 1]", unsqueeze_2018: "f32[1, 432, 1, 1]", le_102: "b8[8, 432, 42, 42]", unsqueeze_2030: "f32[1, 432, 1, 1]", unsqueeze_2042: "f32[1, 432, 1, 1]", unsqueeze_2054: "f32[1, 432, 1, 1]", unsqueeze_2066: "f32[1, 432, 1, 1]", unsqueeze_2078: "f32[1, 432, 1, 1]", unsqueeze_2090: "f32[1, 432, 1, 1]", unsqueeze_2102: "f32[1, 432, 1, 1]", unsqueeze_2114: "f32[1, 432, 1, 1]", unsqueeze_2126: "f32[1, 432, 1, 1]", unsqueeze_2138: "f32[1, 432, 1, 1]", unsqueeze_2150: "f32[1, 432, 1, 1]", unsqueeze_2162: "f32[1, 432, 1, 1]", unsqueeze_2174: "f32[1, 216, 1, 1]", unsqueeze_2186: "f32[1, 216, 1, 1]", unsqueeze_2198: "f32[1, 216, 1, 1]", unsqueeze_2210: "f32[1, 216, 1, 1]", unsqueeze_2222: "f32[1, 216, 1, 1]", unsqueeze_2234: "f32[1, 216, 1, 1]", unsqueeze_2246: "f32[1, 216, 1, 1]", unsqueeze_2258: "f32[1, 216, 1, 1]", unsqueeze_2270: "f32[1, 216, 1, 1]", unsqueeze_2282: "f32[1, 216, 1, 1]", unsqueeze_2294: "f32[1, 216, 1, 1]", unsqueeze_2306: "f32[1, 216, 1, 1]", unsqueeze_2318: "f32[1, 216, 1, 1]", unsqueeze_2330: "f32[1, 216, 1, 1]", unsqueeze_2342: "f32[1, 216, 1, 1]", unsqueeze_2354: "f32[1, 216, 1, 1]", unsqueeze_2366: "f32[1, 216, 1, 1]", unsqueeze_2378: "f32[1, 216, 1, 1]", unsqueeze_2390: "f32[1, 216, 1, 1]", unsqueeze_2402: "f32[1, 216, 1, 1]", unsqueeze_2414: "f32[1, 216, 1, 1]", unsqueeze_2426: "f32[1, 216, 1, 1]", unsqueeze_2438: "f32[1, 216, 1, 1]", unsqueeze_2450: "f32[1, 216, 1, 1]", unsqueeze_2462: "f32[1, 216, 1, 1]", unsqueeze_2474: "f32[1, 216, 1, 1]", unsqueeze_2486: "f32[1, 216, 1, 1]", unsqueeze_2498: "f32[1, 216, 1, 1]", unsqueeze_2510: "f32[1, 216, 1, 1]", unsqueeze_2522: "f32[1, 216, 1, 1]", unsqueeze_2534: "f32[1, 216, 1, 1]", unsqueeze_2546: "f32[1, 216, 1, 1]", unsqueeze_2558: "f32[1, 216, 1, 1]", unsqueeze_2570: "f32[1, 216, 1, 1]", unsqueeze_2582: "f32[1, 216, 1, 1]", unsqueeze_2594: "f32[1, 216, 1, 1]", unsqueeze_2606: "f32[1, 216, 1, 1]", unsqueeze_2618: "f32[1, 216, 1, 1]", unsqueeze_2630: "f32[1, 216, 1, 1]", unsqueeze_2642: "f32[1, 216, 1, 1]", unsqueeze_2654: "f32[1, 216, 1, 1]", unsqueeze_2666: "f32[1, 216, 1, 1]", unsqueeze_2678: "f32[1, 216, 1, 1]", unsqueeze_2690: "f32[1, 216, 1, 1]", unsqueeze_2702: "f32[1, 216, 1, 1]", unsqueeze_2714: "f32[1, 216, 1, 1]", unsqueeze_2726: "f32[1, 216, 1, 1]", unsqueeze_2738: "f32[1, 216, 1, 1]", unsqueeze_2750: "f32[1, 216, 1, 1]", unsqueeze_2762: "f32[1, 216, 1, 1]", unsqueeze_2774: "f32[1, 216, 1, 1]", unsqueeze_2786: "f32[1, 216, 1, 1]", unsqueeze_2798: "f32[1, 216, 1, 1]", unsqueeze_2810: "f32[1, 216, 1, 1]", unsqueeze_2822: "f32[1, 216, 1, 1]", unsqueeze_2834: "f32[1, 216, 1, 1]", unsqueeze_2846: "f32[1, 108, 1, 1]", le_171: "b8[8, 108, 83, 83]", unsqueeze_2858: "f32[1, 108, 1, 1]", unsqueeze_2870: "f32[1, 108, 1, 1]", le_173: "b8[8, 108, 83, 83]", unsqueeze_2882: "f32[1, 108, 1, 1]", unsqueeze_2894: "f32[1, 108, 1, 1]", unsqueeze_2906: "f32[1, 108, 1, 1]", unsqueeze_2918: "f32[1, 108, 1, 1]", unsqueeze_2930: "f32[1, 108, 1, 1]", unsqueeze_2942: "f32[1, 108, 1, 1]", unsqueeze_2954: "f32[1, 108, 1, 1]", unsqueeze_2966: "f32[1, 108, 1, 1]", unsqueeze_2978: "f32[1, 108, 1, 1]", unsqueeze_2990: "f32[1, 108, 1, 1]", unsqueeze_3002: "f32[1, 108, 1, 1]", unsqueeze_3014: "f32[1, 108, 1, 1]", unsqueeze_3026: "f32[1, 54, 1, 1]", le_186: "b8[8, 54, 165, 165]", unsqueeze_3038: "f32[1, 54, 1, 1]", unsqueeze_3050: "f32[1, 54, 1, 1]", unsqueeze_3062: "f32[1, 54, 1, 1]", unsqueeze_3074: "f32[1, 54, 1, 1]", unsqueeze_3086: "f32[1, 54, 1, 1]", unsqueeze_3098: "f32[1, 54, 1, 1]", unsqueeze_3110: "f32[1, 54, 1, 1]", unsqueeze_3122: "f32[1, 54, 1, 1]", unsqueeze_3134: "f32[1, 54, 1, 1]", unsqueeze_3146: "f32[1, 54, 1, 1]", unsqueeze_3158: "f32[1, 54, 1, 1]", unsqueeze_3170: "f32[1, 54, 1, 1]", unsqueeze_3182: "f32[1, 54, 1, 1]", unsqueeze_3194: "f32[1, 54, 1, 1]", unsqueeze_3206: "f32[1, 96, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:337, code: return x if pre_logits else self.last_linear(x)
    mm: "f32[8, 4320]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 4320]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[4320, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 4320]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 4320, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 4320, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 4320, 11, 11]" = torch.ops.aten.expand.default(view_2, [8, 4320, 11, 11]);  view_2 = None
    div: "f32[8, 4320, 11, 11]" = torch.ops.aten.div.Scalar(expand, 121);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:331, code: x = self.act(x_cell_11)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_1: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where, 1, 0, 864)
    slice_2: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where, 1, 864, 1728)
    slice_3: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where, 1, 1728, 2592)
    slice_4: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where, 1, 2592, 3456)
    slice_5: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where, 1, 3456, 4320);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_2: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_5, [0, 2, 3])
    sub_201: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_372, unsqueeze_806);  convolution_372 = unsqueeze_806 = None
    mul_1407: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_5, sub_201)
    sum_3: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1407, [0, 2, 3]);  mul_1407 = None
    mul_1408: "f32[864]" = torch.ops.aten.mul.Tensor(sum_2, 0.0010330578512396695)
    unsqueeze_807: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_808: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1409: "f32[864]" = torch.ops.aten.mul.Tensor(sum_3, 0.0010330578512396695)
    mul_1410: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_601, squeeze_601)
    mul_1411: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1409, mul_1410);  mul_1409 = mul_1410 = None
    unsqueeze_810: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_811: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1412: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_601, primals_774);  primals_774 = None
    unsqueeze_813: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1412, 0);  mul_1412 = None
    unsqueeze_814: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    mul_1413: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_812);  sub_201 = unsqueeze_812 = None
    sub_203: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_5, mul_1413);  mul_1413 = None
    sub_204: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_809);  sub_203 = unsqueeze_809 = None
    mul_1414: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_815);  sub_204 = unsqueeze_815 = None
    mul_1415: "f32[864]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_601);  sum_3 = squeeze_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_1414, convolution_371, primals_773, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1414 = convolution_371 = primals_773 = None
    getitem_486: "f32[8, 864, 11, 11]" = convolution_backward[0]
    getitem_487: "f32[864, 864, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_486, relu_198, primals_772, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_486 = primals_772 = None
    getitem_489: "f32[8, 864, 11, 11]" = convolution_backward_1[0]
    getitem_490: "f32[864, 1, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_1: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_198, 0);  relu_198 = None
    where_1: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_1, full_default, getitem_489);  le_1 = getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_4: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_205: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_370, unsqueeze_818);  convolution_370 = unsqueeze_818 = None
    mul_1416: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_1, sub_205)
    sum_5: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1416, [0, 2, 3]);  mul_1416 = None
    mul_1417: "f32[864]" = torch.ops.aten.mul.Tensor(sum_4, 0.0010330578512396695)
    unsqueeze_819: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_820: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1418: "f32[864]" = torch.ops.aten.mul.Tensor(sum_5, 0.0010330578512396695)
    mul_1419: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_598, squeeze_598)
    mul_1420: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1418, mul_1419);  mul_1418 = mul_1419 = None
    unsqueeze_822: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_823: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1421: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_598, primals_770);  primals_770 = None
    unsqueeze_825: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1421, 0);  mul_1421 = None
    unsqueeze_826: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    mul_1422: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_824);  sub_205 = unsqueeze_824 = None
    sub_207: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_1, mul_1422);  where_1 = mul_1422 = None
    sub_208: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_821);  sub_207 = unsqueeze_821 = None
    mul_1423: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_827);  sub_208 = unsqueeze_827 = None
    mul_1424: "f32[864]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_598);  sum_5 = squeeze_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_1423, convolution_369, primals_769, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1423 = convolution_369 = primals_769 = None
    getitem_492: "f32[8, 864, 11, 11]" = convolution_backward_2[0]
    getitem_493: "f32[864, 864, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_492, relu_187, primals_768, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_492 = primals_768 = None
    getitem_495: "f32[8, 864, 11, 11]" = convolution_backward_3[0]
    getitem_496: "f32[864, 1, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_2: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_187, 0)
    where_2: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_2, full_default, getitem_495);  getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_4, add_1009, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_467)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1075: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_5, max_pool2d_with_indices_backward);  slice_5 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_6: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_4, [0, 2, 3])
    sub_209: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_368, unsqueeze_830);  convolution_368 = unsqueeze_830 = None
    mul_1425: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_4, sub_209)
    sum_7: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1425, [0, 2, 3]);  mul_1425 = None
    mul_1426: "f32[864]" = torch.ops.aten.mul.Tensor(sum_6, 0.0010330578512396695)
    unsqueeze_831: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_832: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1427: "f32[864]" = torch.ops.aten.mul.Tensor(sum_7, 0.0010330578512396695)
    mul_1428: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_595, squeeze_595)
    mul_1429: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1427, mul_1428);  mul_1427 = mul_1428 = None
    unsqueeze_834: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_835: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1430: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_595, primals_766);  primals_766 = None
    unsqueeze_837: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1430, 0);  mul_1430 = None
    unsqueeze_838: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_1431: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_836);  sub_209 = unsqueeze_836 = None
    sub_211: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_4, mul_1431);  slice_4 = mul_1431 = None
    sub_212: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_833);  sub_211 = unsqueeze_833 = None
    mul_1432: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_839);  sub_212 = unsqueeze_839 = None
    mul_1433: "f32[864]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_595);  sum_7 = squeeze_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_1432, convolution_367, primals_765, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1432 = convolution_367 = primals_765 = None
    getitem_498: "f32[8, 864, 11, 11]" = convolution_backward_4[0]
    getitem_499: "f32[864, 864, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_498, relu_196, primals_764, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_498 = primals_764 = None
    getitem_501: "f32[8, 864, 11, 11]" = convolution_backward_5[0]
    getitem_502: "f32[864, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_3: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_196, 0);  relu_196 = None
    where_3: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_3, full_default, getitem_501);  le_3 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_8: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_213: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_366, unsqueeze_842);  convolution_366 = unsqueeze_842 = None
    mul_1434: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_3, sub_213)
    sum_9: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1434, [0, 2, 3]);  mul_1434 = None
    mul_1435: "f32[864]" = torch.ops.aten.mul.Tensor(sum_8, 0.0010330578512396695)
    unsqueeze_843: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_844: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1436: "f32[864]" = torch.ops.aten.mul.Tensor(sum_9, 0.0010330578512396695)
    mul_1437: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_592, squeeze_592)
    mul_1438: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1436, mul_1437);  mul_1436 = mul_1437 = None
    unsqueeze_846: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_847: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1439: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_592, primals_762);  primals_762 = None
    unsqueeze_849: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
    unsqueeze_850: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    mul_1440: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_848);  sub_213 = unsqueeze_848 = None
    sub_215: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_3, mul_1440);  where_3 = mul_1440 = None
    sub_216: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_845);  sub_215 = unsqueeze_845 = None
    mul_1441: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_851);  sub_216 = unsqueeze_851 = None
    mul_1442: "f32[864]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_592);  sum_9 = squeeze_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_1441, convolution_365, primals_761, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1441 = convolution_365 = primals_761 = None
    getitem_504: "f32[8, 864, 11, 11]" = convolution_backward_6[0]
    getitem_505: "f32[864, 864, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_504, relu_195, primals_760, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_504 = primals_760 = None
    getitem_507: "f32[8, 864, 11, 11]" = convolution_backward_7[0]
    getitem_508: "f32[864, 1, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_4: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_195, 0);  relu_195 = None
    where_4: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_4, full_default, getitem_507);  le_4 = getitem_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1076: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_3, where_4);  slice_3 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_10: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1076, [0, 2, 3])
    sub_217: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_364, unsqueeze_854);  convolution_364 = unsqueeze_854 = None
    mul_1443: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1076, sub_217)
    sum_11: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1443, [0, 2, 3]);  mul_1443 = None
    mul_1444: "f32[864]" = torch.ops.aten.mul.Tensor(sum_10, 0.0010330578512396695)
    unsqueeze_855: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_856: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1445: "f32[864]" = torch.ops.aten.mul.Tensor(sum_11, 0.0010330578512396695)
    mul_1446: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_589, squeeze_589)
    mul_1447: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1445, mul_1446);  mul_1445 = mul_1446 = None
    unsqueeze_858: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_859: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1448: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_589, primals_758);  primals_758 = None
    unsqueeze_861: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1448, 0);  mul_1448 = None
    unsqueeze_862: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    mul_1449: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_860);  sub_217 = unsqueeze_860 = None
    sub_219: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1076, mul_1449);  mul_1449 = None
    sub_220: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_857);  sub_219 = None
    mul_1450: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_863);  sub_220 = unsqueeze_863 = None
    mul_1451: "f32[864]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_589);  sum_11 = squeeze_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_1450, convolution_363, primals_757, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1450 = convolution_363 = primals_757 = None
    getitem_510: "f32[8, 864, 11, 11]" = convolution_backward_8[0]
    getitem_511: "f32[864, 864, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_510, relu_194, primals_756, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_510 = primals_756 = None
    getitem_513: "f32[8, 864, 11, 11]" = convolution_backward_9[0]
    getitem_514: "f32[864, 1, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_5: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_194, 0);  relu_194 = None
    where_5: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_5, full_default, getitem_513);  le_5 = getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_12: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_221: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_362, unsqueeze_866);  convolution_362 = unsqueeze_866 = None
    mul_1452: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_5, sub_221)
    sum_13: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1452, [0, 2, 3]);  mul_1452 = None
    mul_1453: "f32[864]" = torch.ops.aten.mul.Tensor(sum_12, 0.0010330578512396695)
    unsqueeze_867: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1453, 0);  mul_1453 = None
    unsqueeze_868: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1454: "f32[864]" = torch.ops.aten.mul.Tensor(sum_13, 0.0010330578512396695)
    mul_1455: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_586, squeeze_586)
    mul_1456: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1454, mul_1455);  mul_1454 = mul_1455 = None
    unsqueeze_870: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_871: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1457: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_586, primals_754);  primals_754 = None
    unsqueeze_873: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1457, 0);  mul_1457 = None
    unsqueeze_874: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    mul_1458: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_872);  sub_221 = unsqueeze_872 = None
    sub_223: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_5, mul_1458);  where_5 = mul_1458 = None
    sub_224: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_869);  sub_223 = unsqueeze_869 = None
    mul_1459: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_875);  sub_224 = unsqueeze_875 = None
    mul_1460: "f32[864]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_586);  sum_13 = squeeze_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_1459, convolution_361, primals_753, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1459 = convolution_361 = primals_753 = None
    getitem_516: "f32[8, 864, 11, 11]" = convolution_backward_10[0]
    getitem_517: "f32[864, 864, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_516, relu_189, primals_752, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_516 = primals_752 = None
    getitem_519: "f32[8, 864, 11, 11]" = convolution_backward_11[0]
    getitem_520: "f32[864, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_6: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_189, 0)
    where_6: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_6, full_default, getitem_519);  getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1077: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1075, where_6);  add_1075 = where_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_225: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_360, unsqueeze_878);  convolution_360 = unsqueeze_878 = None
    mul_1461: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1076, sub_225)
    sum_15: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1461, [0, 2, 3]);  mul_1461 = None
    mul_1463: "f32[864]" = torch.ops.aten.mul.Tensor(sum_15, 0.0010330578512396695)
    mul_1464: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_583, squeeze_583)
    mul_1465: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1463, mul_1464);  mul_1463 = mul_1464 = None
    unsqueeze_882: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_883: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1466: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_583, primals_750);  primals_750 = None
    unsqueeze_885: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1466, 0);  mul_1466 = None
    unsqueeze_886: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    mul_1467: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_884);  sub_225 = unsqueeze_884 = None
    sub_227: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1076, mul_1467);  add_1076 = mul_1467 = None
    sub_228: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_857);  sub_227 = unsqueeze_857 = None
    mul_1468: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_887);  sub_228 = unsqueeze_887 = None
    mul_1469: "f32[864]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_583);  sum_15 = squeeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_1468, convolution_359, primals_749, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1468 = convolution_359 = primals_749 = None
    getitem_522: "f32[8, 864, 11, 11]" = convolution_backward_12[0]
    getitem_523: "f32[864, 864, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_522, relu_192, primals_748, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_522 = primals_748 = None
    getitem_525: "f32[8, 864, 11, 11]" = convolution_backward_13[0]
    getitem_526: "f32[864, 1, 5, 5]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_7: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_192, 0);  relu_192 = None
    where_7: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_7, full_default, getitem_525);  le_7 = getitem_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_16: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_229: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_358, unsqueeze_890);  convolution_358 = unsqueeze_890 = None
    mul_1470: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_7, sub_229)
    sum_17: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1470, [0, 2, 3]);  mul_1470 = None
    mul_1471: "f32[864]" = torch.ops.aten.mul.Tensor(sum_16, 0.0010330578512396695)
    unsqueeze_891: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1471, 0);  mul_1471 = None
    unsqueeze_892: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1472: "f32[864]" = torch.ops.aten.mul.Tensor(sum_17, 0.0010330578512396695)
    mul_1473: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_580, squeeze_580)
    mul_1474: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1472, mul_1473);  mul_1472 = mul_1473 = None
    unsqueeze_894: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_895: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1475: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_580, primals_746);  primals_746 = None
    unsqueeze_897: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
    unsqueeze_898: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_1476: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_896);  sub_229 = unsqueeze_896 = None
    sub_231: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_7, mul_1476);  where_7 = mul_1476 = None
    sub_232: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_893);  sub_231 = unsqueeze_893 = None
    mul_1477: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_899);  sub_232 = unsqueeze_899 = None
    mul_1478: "f32[864]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_580);  sum_17 = squeeze_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_1477, convolution_357, primals_745, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1477 = convolution_357 = primals_745 = None
    getitem_528: "f32[8, 864, 11, 11]" = convolution_backward_14[0]
    getitem_529: "f32[864, 864, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_528, relu_189, primals_744, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_528 = primals_744 = None
    getitem_531: "f32[8, 864, 11, 11]" = convolution_backward_15[0]
    getitem_532: "f32[864, 1, 5, 5]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_8: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_6, full_default, getitem_531);  getitem_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1078: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1077, where_8);  add_1077 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_1: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_2, add_1009, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_467);  add_1009 = getitem_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1079: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1078, max_pool2d_with_indices_backward_1);  add_1078 = max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_18: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_2, [0, 2, 3])
    sub_233: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_356, unsqueeze_902);  convolution_356 = unsqueeze_902 = None
    mul_1479: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_2, sub_233)
    sum_19: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1479, [0, 2, 3]);  mul_1479 = None
    mul_1480: "f32[864]" = torch.ops.aten.mul.Tensor(sum_18, 0.0010330578512396695)
    unsqueeze_903: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1480, 0);  mul_1480 = None
    unsqueeze_904: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1481: "f32[864]" = torch.ops.aten.mul.Tensor(sum_19, 0.0010330578512396695)
    mul_1482: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_577, squeeze_577)
    mul_1483: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1481, mul_1482);  mul_1481 = mul_1482 = None
    unsqueeze_906: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1483, 0);  mul_1483 = None
    unsqueeze_907: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1484: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_577, primals_742);  primals_742 = None
    unsqueeze_909: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
    unsqueeze_910: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_1485: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_908);  sub_233 = unsqueeze_908 = None
    sub_235: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_2, mul_1485);  slice_2 = mul_1485 = None
    sub_236: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_905);  sub_235 = unsqueeze_905 = None
    mul_1486: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_911);  sub_236 = unsqueeze_911 = None
    mul_1487: "f32[864]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_577);  sum_19 = squeeze_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_1486, convolution_355, primals_741, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1486 = convolution_355 = primals_741 = None
    getitem_534: "f32[8, 864, 11, 11]" = convolution_backward_16[0]
    getitem_535: "f32[864, 864, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_534, relu_190, primals_740, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_534 = primals_740 = None
    getitem_537: "f32[8, 864, 11, 11]" = convolution_backward_17[0]
    getitem_538: "f32[864, 1, 7, 7]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_9: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_190, 0);  relu_190 = None
    where_9: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_9, full_default, getitem_537);  le_9 = getitem_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_20: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_237: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_354, unsqueeze_914);  convolution_354 = unsqueeze_914 = None
    mul_1488: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_9, sub_237)
    sum_21: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1488, [0, 2, 3]);  mul_1488 = None
    mul_1489: "f32[864]" = torch.ops.aten.mul.Tensor(sum_20, 0.0010330578512396695)
    unsqueeze_915: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1489, 0);  mul_1489 = None
    unsqueeze_916: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1490: "f32[864]" = torch.ops.aten.mul.Tensor(sum_21, 0.0010330578512396695)
    mul_1491: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_574, squeeze_574)
    mul_1492: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1490, mul_1491);  mul_1490 = mul_1491 = None
    unsqueeze_918: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_919: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1493: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_574, primals_738);  primals_738 = None
    unsqueeze_921: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1493, 0);  mul_1493 = None
    unsqueeze_922: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    mul_1494: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_920);  sub_237 = unsqueeze_920 = None
    sub_239: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_9, mul_1494);  where_9 = mul_1494 = None
    sub_240: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_917);  sub_239 = unsqueeze_917 = None
    mul_1495: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_923);  sub_240 = unsqueeze_923 = None
    mul_1496: "f32[864]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_574);  sum_21 = squeeze_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_1495, convolution_353, primals_737, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1495 = convolution_353 = primals_737 = None
    getitem_540: "f32[8, 864, 11, 11]" = convolution_backward_18[0]
    getitem_541: "f32[864, 864, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_540, relu_189, primals_736, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_540 = relu_189 = primals_736 = None
    getitem_543: "f32[8, 864, 11, 11]" = convolution_backward_19[0]
    getitem_544: "f32[864, 1, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_10: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_6, full_default, getitem_543);  le_6 = getitem_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1080: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1079, where_10);  add_1079 = where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_2: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_1, add_1004, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_461);  add_1004 = getitem_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1081: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(where_2, max_pool2d_with_indices_backward_2);  where_2 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_22: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_1, [0, 2, 3])
    sub_241: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_352, unsqueeze_926);  convolution_352 = unsqueeze_926 = None
    mul_1497: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_1, sub_241)
    sum_23: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1497, [0, 2, 3]);  mul_1497 = None
    mul_1498: "f32[864]" = torch.ops.aten.mul.Tensor(sum_22, 0.0010330578512396695)
    unsqueeze_927: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_928: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1499: "f32[864]" = torch.ops.aten.mul.Tensor(sum_23, 0.0010330578512396695)
    mul_1500: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_571, squeeze_571)
    mul_1501: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1499, mul_1500);  mul_1499 = mul_1500 = None
    unsqueeze_930: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1501, 0);  mul_1501 = None
    unsqueeze_931: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1502: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_571, primals_734);  primals_734 = None
    unsqueeze_933: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1502, 0);  mul_1502 = None
    unsqueeze_934: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    mul_1503: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_932);  sub_241 = unsqueeze_932 = None
    sub_243: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_1, mul_1503);  slice_1 = mul_1503 = None
    sub_244: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_929);  sub_243 = unsqueeze_929 = None
    mul_1504: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_935);  sub_244 = unsqueeze_935 = None
    mul_1505: "f32[864]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_571);  sum_23 = squeeze_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_1504, convolution_351, primals_733, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1504 = convolution_351 = primals_733 = None
    getitem_546: "f32[8, 864, 11, 11]" = convolution_backward_20[0]
    getitem_547: "f32[864, 864, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_546, relu_188, primals_732, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_546 = primals_732 = None
    getitem_549: "f32[8, 864, 11, 11]" = convolution_backward_21[0]
    getitem_550: "f32[864, 1, 5, 5]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_11: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_188, 0);  relu_188 = None
    where_11: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_11, full_default, getitem_549);  le_11 = getitem_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_24: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_245: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_350, unsqueeze_938);  convolution_350 = unsqueeze_938 = None
    mul_1506: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_11, sub_245)
    sum_25: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1506, [0, 2, 3]);  mul_1506 = None
    mul_1507: "f32[864]" = torch.ops.aten.mul.Tensor(sum_24, 0.0010330578512396695)
    unsqueeze_939: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_940: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1508: "f32[864]" = torch.ops.aten.mul.Tensor(sum_25, 0.0010330578512396695)
    mul_1509: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_568, squeeze_568)
    mul_1510: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1508, mul_1509);  mul_1508 = mul_1509 = None
    unsqueeze_942: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_943: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1511: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_568, primals_730);  primals_730 = None
    unsqueeze_945: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1511, 0);  mul_1511 = None
    unsqueeze_946: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    mul_1512: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_944);  sub_245 = unsqueeze_944 = None
    sub_247: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_11, mul_1512);  where_11 = mul_1512 = None
    sub_248: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_941);  sub_247 = unsqueeze_941 = None
    mul_1513: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_947);  sub_248 = unsqueeze_947 = None
    mul_1514: "f32[864]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_568);  sum_25 = squeeze_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_1513, convolution_349, primals_729, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1513 = convolution_349 = primals_729 = None
    getitem_552: "f32[8, 864, 11, 11]" = convolution_backward_22[0]
    getitem_553: "f32[864, 864, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_552, relu_187, primals_728, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_552 = relu_187 = primals_728 = None
    getitem_555: "f32[8, 864, 11, 11]" = convolution_backward_23[0]
    getitem_556: "f32[864, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_12: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_2, full_default, getitem_555);  le_2 = getitem_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1082: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1081, where_12);  add_1081 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_26: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1080, [0, 2, 3])
    sub_249: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_348, unsqueeze_950);  convolution_348 = unsqueeze_950 = None
    mul_1515: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1080, sub_249)
    sum_27: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1515, [0, 2, 3]);  mul_1515 = None
    mul_1516: "f32[864]" = torch.ops.aten.mul.Tensor(sum_26, 0.0010330578512396695)
    unsqueeze_951: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_952: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1517: "f32[864]" = torch.ops.aten.mul.Tensor(sum_27, 0.0010330578512396695)
    mul_1518: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_565, squeeze_565)
    mul_1519: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1517, mul_1518);  mul_1517 = mul_1518 = None
    unsqueeze_954: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1519, 0);  mul_1519 = None
    unsqueeze_955: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1520: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_565, primals_726);  primals_726 = None
    unsqueeze_957: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1520, 0);  mul_1520 = None
    unsqueeze_958: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    mul_1521: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_956);  sub_249 = unsqueeze_956 = None
    sub_251: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1080, mul_1521);  add_1080 = mul_1521 = None
    sub_252: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_953);  sub_251 = unsqueeze_953 = None
    mul_1522: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_959);  sub_252 = unsqueeze_959 = None
    mul_1523: "f32[864]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_565);  sum_27 = squeeze_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_1522, relu_186, primals_725, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1522 = primals_725 = None
    getitem_558: "f32[8, 4320, 11, 11]" = convolution_backward_24[0]
    getitem_559: "f32[864, 4320, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_13: "b8[8, 4320, 11, 11]" = torch.ops.aten.le.Scalar(relu_186, 0);  relu_186 = None
    where_13: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le_13, full_default, getitem_558);  le_13 = getitem_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_28: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1082, [0, 2, 3])
    sub_253: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_347, unsqueeze_962);  convolution_347 = unsqueeze_962 = None
    mul_1524: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1082, sub_253)
    sum_29: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1524, [0, 2, 3]);  mul_1524 = None
    mul_1525: "f32[864]" = torch.ops.aten.mul.Tensor(sum_28, 0.0010330578512396695)
    unsqueeze_963: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1525, 0);  mul_1525 = None
    unsqueeze_964: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1526: "f32[864]" = torch.ops.aten.mul.Tensor(sum_29, 0.0010330578512396695)
    mul_1527: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_562, squeeze_562)
    mul_1528: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1526, mul_1527);  mul_1526 = mul_1527 = None
    unsqueeze_966: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_967: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1529: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_562, primals_723);  primals_723 = None
    unsqueeze_969: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1529, 0);  mul_1529 = None
    unsqueeze_970: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    mul_1530: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_968);  sub_253 = unsqueeze_968 = None
    sub_255: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1082, mul_1530);  add_1082 = mul_1530 = None
    sub_256: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_965);  sub_255 = unsqueeze_965 = None
    mul_1531: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_971);  sub_256 = unsqueeze_971 = None
    mul_1532: "f32[864]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_562);  sum_29 = squeeze_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1531, relu_172, primals_722, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1531 = primals_722 = None
    getitem_561: "f32[8, 4320, 11, 11]" = convolution_backward_25[0]
    getitem_562: "f32[864, 4320, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_14: "b8[8, 4320, 11, 11]" = torch.ops.aten.le.Scalar(relu_172, 0)
    where_14: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le_14, full_default, getitem_561);  getitem_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_6: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where_13, 1, 0, 864)
    slice_7: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where_13, 1, 864, 1728)
    slice_8: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where_13, 1, 1728, 2592)
    slice_9: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where_13, 1, 2592, 3456)
    slice_10: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(where_13, 1, 3456, 4320);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_30: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_10, [0, 2, 3])
    sub_257: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_346, unsqueeze_974);  convolution_346 = unsqueeze_974 = None
    mul_1533: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_10, sub_257)
    sum_31: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1533, [0, 2, 3]);  mul_1533 = None
    mul_1534: "f32[864]" = torch.ops.aten.mul.Tensor(sum_30, 0.0010330578512396695)
    unsqueeze_975: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1534, 0);  mul_1534 = None
    unsqueeze_976: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1535: "f32[864]" = torch.ops.aten.mul.Tensor(sum_31, 0.0010330578512396695)
    mul_1536: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_559, squeeze_559)
    mul_1537: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1535, mul_1536);  mul_1535 = mul_1536 = None
    unsqueeze_978: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_979: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1538: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_559, primals_720);  primals_720 = None
    unsqueeze_981: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1538, 0);  mul_1538 = None
    unsqueeze_982: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    mul_1539: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_980);  sub_257 = unsqueeze_980 = None
    sub_259: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_10, mul_1539);  mul_1539 = None
    sub_260: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_977);  sub_259 = unsqueeze_977 = None
    mul_1540: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_983);  sub_260 = unsqueeze_983 = None
    mul_1541: "f32[864]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_559);  sum_31 = squeeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_1540, convolution_345, primals_719, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1540 = convolution_345 = primals_719 = None
    getitem_564: "f32[8, 864, 11, 11]" = convolution_backward_26[0]
    getitem_565: "f32[864, 864, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(getitem_564, relu_184, primals_718, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_564 = primals_718 = None
    getitem_567: "f32[8, 864, 11, 11]" = convolution_backward_27[0]
    getitem_568: "f32[864, 1, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_15: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_184, 0);  relu_184 = None
    where_15: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_15, full_default, getitem_567);  le_15 = getitem_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_32: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_261: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_344, unsqueeze_986);  convolution_344 = unsqueeze_986 = None
    mul_1542: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_15, sub_261)
    sum_33: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1542, [0, 2, 3]);  mul_1542 = None
    mul_1543: "f32[864]" = torch.ops.aten.mul.Tensor(sum_32, 0.0010330578512396695)
    unsqueeze_987: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1543, 0);  mul_1543 = None
    unsqueeze_988: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1544: "f32[864]" = torch.ops.aten.mul.Tensor(sum_33, 0.0010330578512396695)
    mul_1545: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_556, squeeze_556)
    mul_1546: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1544, mul_1545);  mul_1544 = mul_1545 = None
    unsqueeze_990: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_991: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1547: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_556, primals_716);  primals_716 = None
    unsqueeze_993: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1547, 0);  mul_1547 = None
    unsqueeze_994: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    mul_1548: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_992);  sub_261 = unsqueeze_992 = None
    sub_263: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_15, mul_1548);  where_15 = mul_1548 = None
    sub_264: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_989);  sub_263 = unsqueeze_989 = None
    mul_1549: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_995);  sub_264 = unsqueeze_995 = None
    mul_1550: "f32[864]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_556);  sum_33 = squeeze_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_1549, convolution_343, primals_715, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1549 = convolution_343 = primals_715 = None
    getitem_570: "f32[8, 864, 11, 11]" = convolution_backward_28[0]
    getitem_571: "f32[864, 864, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(getitem_570, relu_173, primals_714, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_570 = primals_714 = None
    getitem_573: "f32[8, 864, 11, 11]" = convolution_backward_29[0]
    getitem_574: "f32[864, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_16: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_173, 0)
    where_16: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_16, full_default, getitem_573);  getitem_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_3: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_9, add_934, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_433)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1083: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_10, max_pool2d_with_indices_backward_3);  slice_10 = max_pool2d_with_indices_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_34: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_9, [0, 2, 3])
    sub_265: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_342, unsqueeze_998);  convolution_342 = unsqueeze_998 = None
    mul_1551: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_9, sub_265)
    sum_35: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1551, [0, 2, 3]);  mul_1551 = None
    mul_1552: "f32[864]" = torch.ops.aten.mul.Tensor(sum_34, 0.0010330578512396695)
    unsqueeze_999: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1000: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1553: "f32[864]" = torch.ops.aten.mul.Tensor(sum_35, 0.0010330578512396695)
    mul_1554: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_553, squeeze_553)
    mul_1555: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1553, mul_1554);  mul_1553 = mul_1554 = None
    unsqueeze_1002: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1003: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1556: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_553, primals_712);  primals_712 = None
    unsqueeze_1005: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1556, 0);  mul_1556 = None
    unsqueeze_1006: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    mul_1557: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_1004);  sub_265 = unsqueeze_1004 = None
    sub_267: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_9, mul_1557);  slice_9 = mul_1557 = None
    sub_268: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_1001);  sub_267 = unsqueeze_1001 = None
    mul_1558: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_1007);  sub_268 = unsqueeze_1007 = None
    mul_1559: "f32[864]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_553);  sum_35 = squeeze_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1558, convolution_341, primals_711, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1558 = convolution_341 = primals_711 = None
    getitem_576: "f32[8, 864, 11, 11]" = convolution_backward_30[0]
    getitem_577: "f32[864, 864, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(getitem_576, relu_182, primals_710, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_576 = primals_710 = None
    getitem_579: "f32[8, 864, 11, 11]" = convolution_backward_31[0]
    getitem_580: "f32[864, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_17: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_182, 0);  relu_182 = None
    where_17: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_17, full_default, getitem_579);  le_17 = getitem_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_36: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_269: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_340, unsqueeze_1010);  convolution_340 = unsqueeze_1010 = None
    mul_1560: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_17, sub_269)
    sum_37: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1560, [0, 2, 3]);  mul_1560 = None
    mul_1561: "f32[864]" = torch.ops.aten.mul.Tensor(sum_36, 0.0010330578512396695)
    unsqueeze_1011: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1012: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1562: "f32[864]" = torch.ops.aten.mul.Tensor(sum_37, 0.0010330578512396695)
    mul_1563: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_550, squeeze_550)
    mul_1564: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1562, mul_1563);  mul_1562 = mul_1563 = None
    unsqueeze_1014: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1564, 0);  mul_1564 = None
    unsqueeze_1015: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1565: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_550, primals_708);  primals_708 = None
    unsqueeze_1017: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1565, 0);  mul_1565 = None
    unsqueeze_1018: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    mul_1566: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_1016);  sub_269 = unsqueeze_1016 = None
    sub_271: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_17, mul_1566);  where_17 = mul_1566 = None
    sub_272: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_1013);  sub_271 = unsqueeze_1013 = None
    mul_1567: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_1019);  sub_272 = unsqueeze_1019 = None
    mul_1568: "f32[864]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_550);  sum_37 = squeeze_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1567, convolution_339, primals_707, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1567 = convolution_339 = primals_707 = None
    getitem_582: "f32[8, 864, 11, 11]" = convolution_backward_32[0]
    getitem_583: "f32[864, 864, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(getitem_582, relu_181, primals_706, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_582 = primals_706 = None
    getitem_585: "f32[8, 864, 11, 11]" = convolution_backward_33[0]
    getitem_586: "f32[864, 1, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_18: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_181, 0);  relu_181 = None
    where_18: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_18, full_default, getitem_585);  le_18 = getitem_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1084: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_8, where_18);  slice_8 = where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_38: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1084, [0, 2, 3])
    sub_273: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_338, unsqueeze_1022);  convolution_338 = unsqueeze_1022 = None
    mul_1569: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1084, sub_273)
    sum_39: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1569, [0, 2, 3]);  mul_1569 = None
    mul_1570: "f32[864]" = torch.ops.aten.mul.Tensor(sum_38, 0.0010330578512396695)
    unsqueeze_1023: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1024: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1571: "f32[864]" = torch.ops.aten.mul.Tensor(sum_39, 0.0010330578512396695)
    mul_1572: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_547, squeeze_547)
    mul_1573: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1571, mul_1572);  mul_1571 = mul_1572 = None
    unsqueeze_1026: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1573, 0);  mul_1573 = None
    unsqueeze_1027: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1574: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_547, primals_704);  primals_704 = None
    unsqueeze_1029: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1574, 0);  mul_1574 = None
    unsqueeze_1030: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    mul_1575: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_1028);  sub_273 = unsqueeze_1028 = None
    sub_275: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1084, mul_1575);  mul_1575 = None
    sub_276: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_1025);  sub_275 = None
    mul_1576: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_1031);  sub_276 = unsqueeze_1031 = None
    mul_1577: "f32[864]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_547);  sum_39 = squeeze_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1576, convolution_337, primals_703, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1576 = convolution_337 = primals_703 = None
    getitem_588: "f32[8, 864, 11, 11]" = convolution_backward_34[0]
    getitem_589: "f32[864, 864, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(getitem_588, relu_180, primals_702, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_588 = primals_702 = None
    getitem_591: "f32[8, 864, 11, 11]" = convolution_backward_35[0]
    getitem_592: "f32[864, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_19: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_180, 0);  relu_180 = None
    where_19: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_19, full_default, getitem_591);  le_19 = getitem_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_40: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_277: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_336, unsqueeze_1034);  convolution_336 = unsqueeze_1034 = None
    mul_1578: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_19, sub_277)
    sum_41: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1578, [0, 2, 3]);  mul_1578 = None
    mul_1579: "f32[864]" = torch.ops.aten.mul.Tensor(sum_40, 0.0010330578512396695)
    unsqueeze_1035: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1036: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1580: "f32[864]" = torch.ops.aten.mul.Tensor(sum_41, 0.0010330578512396695)
    mul_1581: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_544, squeeze_544)
    mul_1582: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1580, mul_1581);  mul_1580 = mul_1581 = None
    unsqueeze_1038: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1582, 0);  mul_1582 = None
    unsqueeze_1039: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1583: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_544, primals_700);  primals_700 = None
    unsqueeze_1041: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1583, 0);  mul_1583 = None
    unsqueeze_1042: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    mul_1584: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_1040);  sub_277 = unsqueeze_1040 = None
    sub_279: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_19, mul_1584);  where_19 = mul_1584 = None
    sub_280: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_1037);  sub_279 = unsqueeze_1037 = None
    mul_1585: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_1043);  sub_280 = unsqueeze_1043 = None
    mul_1586: "f32[864]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_544);  sum_41 = squeeze_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1585, convolution_335, primals_699, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1585 = convolution_335 = primals_699 = None
    getitem_594: "f32[8, 864, 11, 11]" = convolution_backward_36[0]
    getitem_595: "f32[864, 864, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(getitem_594, relu_175, primals_698, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_594 = primals_698 = None
    getitem_597: "f32[8, 864, 11, 11]" = convolution_backward_37[0]
    getitem_598: "f32[864, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_20: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_175, 0)
    where_20: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_20, full_default, getitem_597);  getitem_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1085: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1083, where_20);  add_1083 = where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_281: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_334, unsqueeze_1046);  convolution_334 = unsqueeze_1046 = None
    mul_1587: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1084, sub_281)
    sum_43: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1587, [0, 2, 3]);  mul_1587 = None
    mul_1589: "f32[864]" = torch.ops.aten.mul.Tensor(sum_43, 0.0010330578512396695)
    mul_1590: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_541, squeeze_541)
    mul_1591: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1589, mul_1590);  mul_1589 = mul_1590 = None
    unsqueeze_1050: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1591, 0);  mul_1591 = None
    unsqueeze_1051: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1592: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_541, primals_696);  primals_696 = None
    unsqueeze_1053: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1592, 0);  mul_1592 = None
    unsqueeze_1054: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    mul_1593: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_1052);  sub_281 = unsqueeze_1052 = None
    sub_283: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1084, mul_1593);  add_1084 = mul_1593 = None
    sub_284: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_1025);  sub_283 = unsqueeze_1025 = None
    mul_1594: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_1055);  sub_284 = unsqueeze_1055 = None
    mul_1595: "f32[864]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_541);  sum_43 = squeeze_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1594, convolution_333, primals_695, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1594 = convolution_333 = primals_695 = None
    getitem_600: "f32[8, 864, 11, 11]" = convolution_backward_38[0]
    getitem_601: "f32[864, 864, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(getitem_600, relu_178, primals_694, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_600 = primals_694 = None
    getitem_603: "f32[8, 864, 11, 11]" = convolution_backward_39[0]
    getitem_604: "f32[864, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_21: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_178, 0);  relu_178 = None
    where_21: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_21, full_default, getitem_603);  le_21 = getitem_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_44: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_285: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_332, unsqueeze_1058);  convolution_332 = unsqueeze_1058 = None
    mul_1596: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_21, sub_285)
    sum_45: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1596, [0, 2, 3]);  mul_1596 = None
    mul_1597: "f32[864]" = torch.ops.aten.mul.Tensor(sum_44, 0.0010330578512396695)
    unsqueeze_1059: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1597, 0);  mul_1597 = None
    unsqueeze_1060: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1598: "f32[864]" = torch.ops.aten.mul.Tensor(sum_45, 0.0010330578512396695)
    mul_1599: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_538, squeeze_538)
    mul_1600: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1598, mul_1599);  mul_1598 = mul_1599 = None
    unsqueeze_1062: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1063: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1601: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_538, primals_692);  primals_692 = None
    unsqueeze_1065: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1601, 0);  mul_1601 = None
    unsqueeze_1066: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    mul_1602: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_1064);  sub_285 = unsqueeze_1064 = None
    sub_287: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_21, mul_1602);  where_21 = mul_1602 = None
    sub_288: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_1061);  sub_287 = unsqueeze_1061 = None
    mul_1603: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_1067);  sub_288 = unsqueeze_1067 = None
    mul_1604: "f32[864]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_538);  sum_45 = squeeze_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1603, convolution_331, primals_691, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1603 = convolution_331 = primals_691 = None
    getitem_606: "f32[8, 864, 11, 11]" = convolution_backward_40[0]
    getitem_607: "f32[864, 864, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(getitem_606, relu_175, primals_690, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_606 = primals_690 = None
    getitem_609: "f32[8, 864, 11, 11]" = convolution_backward_41[0]
    getitem_610: "f32[864, 1, 5, 5]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_22: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_20, full_default, getitem_609);  getitem_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1086: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1085, where_22);  add_1085 = where_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_4: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_7, add_934, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_433);  add_934 = getitem_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1087: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1086, max_pool2d_with_indices_backward_4);  add_1086 = max_pool2d_with_indices_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_46: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_7, [0, 2, 3])
    sub_289: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_330, unsqueeze_1070);  convolution_330 = unsqueeze_1070 = None
    mul_1605: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_7, sub_289)
    sum_47: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1605, [0, 2, 3]);  mul_1605 = None
    mul_1606: "f32[864]" = torch.ops.aten.mul.Tensor(sum_46, 0.0010330578512396695)
    unsqueeze_1071: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1606, 0);  mul_1606 = None
    unsqueeze_1072: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1607: "f32[864]" = torch.ops.aten.mul.Tensor(sum_47, 0.0010330578512396695)
    mul_1608: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_535, squeeze_535)
    mul_1609: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1607, mul_1608);  mul_1607 = mul_1608 = None
    unsqueeze_1074: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_1075: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1610: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_535, primals_688);  primals_688 = None
    unsqueeze_1077: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1610, 0);  mul_1610 = None
    unsqueeze_1078: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    mul_1611: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_1076);  sub_289 = unsqueeze_1076 = None
    sub_291: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_7, mul_1611);  slice_7 = mul_1611 = None
    sub_292: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_1073);  sub_291 = unsqueeze_1073 = None
    mul_1612: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_1079);  sub_292 = unsqueeze_1079 = None
    mul_1613: "f32[864]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_535);  sum_47 = squeeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1612, convolution_329, primals_687, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1612 = convolution_329 = primals_687 = None
    getitem_612: "f32[8, 864, 11, 11]" = convolution_backward_42[0]
    getitem_613: "f32[864, 864, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(getitem_612, relu_176, primals_686, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_612 = primals_686 = None
    getitem_615: "f32[8, 864, 11, 11]" = convolution_backward_43[0]
    getitem_616: "f32[864, 1, 7, 7]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_23: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_176, 0);  relu_176 = None
    where_23: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_23, full_default, getitem_615);  le_23 = getitem_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_48: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_293: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_328, unsqueeze_1082);  convolution_328 = unsqueeze_1082 = None
    mul_1614: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_23, sub_293)
    sum_49: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1614, [0, 2, 3]);  mul_1614 = None
    mul_1615: "f32[864]" = torch.ops.aten.mul.Tensor(sum_48, 0.0010330578512396695)
    unsqueeze_1083: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1615, 0);  mul_1615 = None
    unsqueeze_1084: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1616: "f32[864]" = torch.ops.aten.mul.Tensor(sum_49, 0.0010330578512396695)
    mul_1617: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_532, squeeze_532)
    mul_1618: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1616, mul_1617);  mul_1616 = mul_1617 = None
    unsqueeze_1086: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1618, 0);  mul_1618 = None
    unsqueeze_1087: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1619: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_532, primals_684);  primals_684 = None
    unsqueeze_1089: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1619, 0);  mul_1619 = None
    unsqueeze_1090: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    mul_1620: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1088);  sub_293 = unsqueeze_1088 = None
    sub_295: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_23, mul_1620);  where_23 = mul_1620 = None
    sub_296: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_1085);  sub_295 = unsqueeze_1085 = None
    mul_1621: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_1091);  sub_296 = unsqueeze_1091 = None
    mul_1622: "f32[864]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_532);  sum_49 = squeeze_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1621, convolution_327, primals_683, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1621 = convolution_327 = primals_683 = None
    getitem_618: "f32[8, 864, 11, 11]" = convolution_backward_44[0]
    getitem_619: "f32[864, 864, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(getitem_618, relu_175, primals_682, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_618 = relu_175 = primals_682 = None
    getitem_621: "f32[8, 864, 11, 11]" = convolution_backward_45[0]
    getitem_622: "f32[864, 1, 7, 7]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_24: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_20, full_default, getitem_621);  le_20 = getitem_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1088: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1087, where_24);  add_1087 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_5: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_6, add_929, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_427);  add_929 = getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1089: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(where_16, max_pool2d_with_indices_backward_5);  where_16 = max_pool2d_with_indices_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_50: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_6, [0, 2, 3])
    sub_297: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_326, unsqueeze_1094);  convolution_326 = unsqueeze_1094 = None
    mul_1623: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_6, sub_297)
    sum_51: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1623, [0, 2, 3]);  mul_1623 = None
    mul_1624: "f32[864]" = torch.ops.aten.mul.Tensor(sum_50, 0.0010330578512396695)
    unsqueeze_1095: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1624, 0);  mul_1624 = None
    unsqueeze_1096: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1625: "f32[864]" = torch.ops.aten.mul.Tensor(sum_51, 0.0010330578512396695)
    mul_1626: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_529, squeeze_529)
    mul_1627: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1625, mul_1626);  mul_1625 = mul_1626 = None
    unsqueeze_1098: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1627, 0);  mul_1627 = None
    unsqueeze_1099: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1628: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_529, primals_680);  primals_680 = None
    unsqueeze_1101: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1628, 0);  mul_1628 = None
    unsqueeze_1102: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    mul_1629: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1100);  sub_297 = unsqueeze_1100 = None
    sub_299: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_6, mul_1629);  slice_6 = mul_1629 = None
    sub_300: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_1097);  sub_299 = unsqueeze_1097 = None
    mul_1630: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1103);  sub_300 = unsqueeze_1103 = None
    mul_1631: "f32[864]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_529);  sum_51 = squeeze_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1630, convolution_325, primals_679, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1630 = convolution_325 = primals_679 = None
    getitem_624: "f32[8, 864, 11, 11]" = convolution_backward_46[0]
    getitem_625: "f32[864, 864, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(getitem_624, relu_174, primals_678, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_624 = primals_678 = None
    getitem_627: "f32[8, 864, 11, 11]" = convolution_backward_47[0]
    getitem_628: "f32[864, 1, 5, 5]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_25: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_174, 0);  relu_174 = None
    where_25: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_25, full_default, getitem_627);  le_25 = getitem_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_52: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_301: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_324, unsqueeze_1106);  convolution_324 = unsqueeze_1106 = None
    mul_1632: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_25, sub_301)
    sum_53: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1632, [0, 2, 3]);  mul_1632 = None
    mul_1633: "f32[864]" = torch.ops.aten.mul.Tensor(sum_52, 0.0010330578512396695)
    unsqueeze_1107: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1108: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1634: "f32[864]" = torch.ops.aten.mul.Tensor(sum_53, 0.0010330578512396695)
    mul_1635: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_526, squeeze_526)
    mul_1636: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1634, mul_1635);  mul_1634 = mul_1635 = None
    unsqueeze_1110: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1636, 0);  mul_1636 = None
    unsqueeze_1111: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1637: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_526, primals_676);  primals_676 = None
    unsqueeze_1113: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1637, 0);  mul_1637 = None
    unsqueeze_1114: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    mul_1638: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1112);  sub_301 = unsqueeze_1112 = None
    sub_303: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_25, mul_1638);  where_25 = mul_1638 = None
    sub_304: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_1109);  sub_303 = unsqueeze_1109 = None
    mul_1639: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1115);  sub_304 = unsqueeze_1115 = None
    mul_1640: "f32[864]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_526);  sum_53 = squeeze_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1639, convolution_323, primals_675, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1639 = convolution_323 = primals_675 = None
    getitem_630: "f32[8, 864, 11, 11]" = convolution_backward_48[0]
    getitem_631: "f32[864, 864, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(getitem_630, relu_173, primals_674, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_630 = relu_173 = primals_674 = None
    getitem_633: "f32[8, 864, 11, 11]" = convolution_backward_49[0]
    getitem_634: "f32[864, 1, 5, 5]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_26: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_16, full_default, getitem_633);  le_16 = getitem_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1090: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1089, where_26);  add_1089 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_54: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1088, [0, 2, 3])
    sub_305: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_322, unsqueeze_1118);  convolution_322 = unsqueeze_1118 = None
    mul_1641: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1088, sub_305)
    sum_55: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1641, [0, 2, 3]);  mul_1641 = None
    mul_1642: "f32[864]" = torch.ops.aten.mul.Tensor(sum_54, 0.0010330578512396695)
    unsqueeze_1119: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1120: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1643: "f32[864]" = torch.ops.aten.mul.Tensor(sum_55, 0.0010330578512396695)
    mul_1644: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_523, squeeze_523)
    mul_1645: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1643, mul_1644);  mul_1643 = mul_1644 = None
    unsqueeze_1122: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1645, 0);  mul_1645 = None
    unsqueeze_1123: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1646: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_523, primals_672);  primals_672 = None
    unsqueeze_1125: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1646, 0);  mul_1646 = None
    unsqueeze_1126: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    mul_1647: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1124);  sub_305 = unsqueeze_1124 = None
    sub_307: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1088, mul_1647);  add_1088 = mul_1647 = None
    sub_308: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1121);  sub_307 = unsqueeze_1121 = None
    mul_1648: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1127);  sub_308 = unsqueeze_1127 = None
    mul_1649: "f32[864]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_523);  sum_55 = squeeze_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1648, relu_172, primals_671, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1648 = relu_172 = primals_671 = None
    getitem_636: "f32[8, 4320, 11, 11]" = convolution_backward_50[0]
    getitem_637: "f32[864, 4320, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_27: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le_14, full_default, getitem_636);  le_14 = getitem_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1091: "f32[8, 4320, 11, 11]" = torch.ops.aten.add.Tensor(where_14, where_27);  where_14 = where_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_56: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1090, [0, 2, 3])
    sub_309: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_321, unsqueeze_1130);  convolution_321 = unsqueeze_1130 = None
    mul_1650: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1090, sub_309)
    sum_57: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1650, [0, 2, 3]);  mul_1650 = None
    mul_1651: "f32[864]" = torch.ops.aten.mul.Tensor(sum_56, 0.0010330578512396695)
    unsqueeze_1131: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1132: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1652: "f32[864]" = torch.ops.aten.mul.Tensor(sum_57, 0.0010330578512396695)
    mul_1653: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_520, squeeze_520)
    mul_1654: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1652, mul_1653);  mul_1652 = mul_1653 = None
    unsqueeze_1134: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1654, 0);  mul_1654 = None
    unsqueeze_1135: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1655: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_520, primals_669);  primals_669 = None
    unsqueeze_1137: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1655, 0);  mul_1655 = None
    unsqueeze_1138: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    mul_1656: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1136);  sub_309 = unsqueeze_1136 = None
    sub_311: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1090, mul_1656);  add_1090 = mul_1656 = None
    sub_312: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1133);  sub_311 = unsqueeze_1133 = None
    mul_1657: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1139);  sub_312 = unsqueeze_1139 = None
    mul_1658: "f32[864]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_520);  sum_57 = squeeze_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1657, relu_158, primals_668, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1657 = primals_668 = None
    getitem_639: "f32[8, 4320, 11, 11]" = convolution_backward_51[0]
    getitem_640: "f32[864, 4320, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_28: "b8[8, 4320, 11, 11]" = torch.ops.aten.le.Scalar(relu_158, 0)
    where_28: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le_28, full_default, getitem_639);  getitem_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_11: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1091, 1, 0, 864)
    slice_12: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1091, 1, 864, 1728)
    slice_13: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1091, 1, 1728, 2592)
    slice_14: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1091, 1, 2592, 3456)
    slice_15: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1091, 1, 3456, 4320);  add_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_58: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_15, [0, 2, 3])
    sub_313: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_320, unsqueeze_1142);  convolution_320 = unsqueeze_1142 = None
    mul_1659: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_15, sub_313)
    sum_59: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1659, [0, 2, 3]);  mul_1659 = None
    mul_1660: "f32[864]" = torch.ops.aten.mul.Tensor(sum_58, 0.0010330578512396695)
    unsqueeze_1143: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1144: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1661: "f32[864]" = torch.ops.aten.mul.Tensor(sum_59, 0.0010330578512396695)
    mul_1662: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_517, squeeze_517)
    mul_1663: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1661, mul_1662);  mul_1661 = mul_1662 = None
    unsqueeze_1146: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1663, 0);  mul_1663 = None
    unsqueeze_1147: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1664: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_517, primals_666);  primals_666 = None
    unsqueeze_1149: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1664, 0);  mul_1664 = None
    unsqueeze_1150: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    mul_1665: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1148);  sub_313 = unsqueeze_1148 = None
    sub_315: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_15, mul_1665);  mul_1665 = None
    sub_316: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1145);  sub_315 = unsqueeze_1145 = None
    mul_1666: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1151);  sub_316 = unsqueeze_1151 = None
    mul_1667: "f32[864]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_517);  sum_59 = squeeze_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1666, convolution_319, primals_665, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1666 = convolution_319 = primals_665 = None
    getitem_642: "f32[8, 864, 11, 11]" = convolution_backward_52[0]
    getitem_643: "f32[864, 864, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(getitem_642, relu_170, primals_664, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_642 = primals_664 = None
    getitem_645: "f32[8, 864, 11, 11]" = convolution_backward_53[0]
    getitem_646: "f32[864, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_29: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_170, 0);  relu_170 = None
    where_29: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_29, full_default, getitem_645);  le_29 = getitem_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_60: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_317: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_318, unsqueeze_1154);  convolution_318 = unsqueeze_1154 = None
    mul_1668: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_29, sub_317)
    sum_61: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1668, [0, 2, 3]);  mul_1668 = None
    mul_1669: "f32[864]" = torch.ops.aten.mul.Tensor(sum_60, 0.0010330578512396695)
    unsqueeze_1155: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1669, 0);  mul_1669 = None
    unsqueeze_1156: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1670: "f32[864]" = torch.ops.aten.mul.Tensor(sum_61, 0.0010330578512396695)
    mul_1671: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_514, squeeze_514)
    mul_1672: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1670, mul_1671);  mul_1670 = mul_1671 = None
    unsqueeze_1158: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1159: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1673: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_514, primals_662);  primals_662 = None
    unsqueeze_1161: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1673, 0);  mul_1673 = None
    unsqueeze_1162: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    mul_1674: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1160);  sub_317 = unsqueeze_1160 = None
    sub_319: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_29, mul_1674);  where_29 = mul_1674 = None
    sub_320: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1157);  sub_319 = unsqueeze_1157 = None
    mul_1675: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1163);  sub_320 = unsqueeze_1163 = None
    mul_1676: "f32[864]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_514);  sum_61 = squeeze_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1675, convolution_317, primals_661, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1675 = convolution_317 = primals_661 = None
    getitem_648: "f32[8, 864, 11, 11]" = convolution_backward_54[0]
    getitem_649: "f32[864, 864, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(getitem_648, relu_159, primals_660, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_648 = primals_660 = None
    getitem_651: "f32[8, 864, 11, 11]" = convolution_backward_55[0]
    getitem_652: "f32[864, 1, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_30: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_159, 0)
    where_30: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_30, full_default, getitem_651);  getitem_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_6: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_14, add_859, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_399)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1092: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_15, max_pool2d_with_indices_backward_6);  slice_15 = max_pool2d_with_indices_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_62: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_14, [0, 2, 3])
    sub_321: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_316, unsqueeze_1166);  convolution_316 = unsqueeze_1166 = None
    mul_1677: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_14, sub_321)
    sum_63: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1677, [0, 2, 3]);  mul_1677 = None
    mul_1678: "f32[864]" = torch.ops.aten.mul.Tensor(sum_62, 0.0010330578512396695)
    unsqueeze_1167: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1678, 0);  mul_1678 = None
    unsqueeze_1168: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1679: "f32[864]" = torch.ops.aten.mul.Tensor(sum_63, 0.0010330578512396695)
    mul_1680: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_511, squeeze_511)
    mul_1681: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1679, mul_1680);  mul_1679 = mul_1680 = None
    unsqueeze_1170: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1681, 0);  mul_1681 = None
    unsqueeze_1171: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1682: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_511, primals_658);  primals_658 = None
    unsqueeze_1173: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1682, 0);  mul_1682 = None
    unsqueeze_1174: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    mul_1683: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1172);  sub_321 = unsqueeze_1172 = None
    sub_323: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_14, mul_1683);  slice_14 = mul_1683 = None
    sub_324: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1169);  sub_323 = unsqueeze_1169 = None
    mul_1684: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1175);  sub_324 = unsqueeze_1175 = None
    mul_1685: "f32[864]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_511);  sum_63 = squeeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1684, convolution_315, primals_657, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1684 = convolution_315 = primals_657 = None
    getitem_654: "f32[8, 864, 11, 11]" = convolution_backward_56[0]
    getitem_655: "f32[864, 864, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(getitem_654, relu_168, primals_656, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_654 = primals_656 = None
    getitem_657: "f32[8, 864, 11, 11]" = convolution_backward_57[0]
    getitem_658: "f32[864, 1, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_31: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_168, 0);  relu_168 = None
    where_31: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_31, full_default, getitem_657);  le_31 = getitem_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_64: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_325: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_314, unsqueeze_1178);  convolution_314 = unsqueeze_1178 = None
    mul_1686: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_31, sub_325)
    sum_65: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1686, [0, 2, 3]);  mul_1686 = None
    mul_1687: "f32[864]" = torch.ops.aten.mul.Tensor(sum_64, 0.0010330578512396695)
    unsqueeze_1179: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1687, 0);  mul_1687 = None
    unsqueeze_1180: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1688: "f32[864]" = torch.ops.aten.mul.Tensor(sum_65, 0.0010330578512396695)
    mul_1689: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_508, squeeze_508)
    mul_1690: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1688, mul_1689);  mul_1688 = mul_1689 = None
    unsqueeze_1182: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1690, 0);  mul_1690 = None
    unsqueeze_1183: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1691: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_508, primals_654);  primals_654 = None
    unsqueeze_1185: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1691, 0);  mul_1691 = None
    unsqueeze_1186: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    mul_1692: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1184);  sub_325 = unsqueeze_1184 = None
    sub_327: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_31, mul_1692);  where_31 = mul_1692 = None
    sub_328: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1181);  sub_327 = unsqueeze_1181 = None
    mul_1693: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1187);  sub_328 = unsqueeze_1187 = None
    mul_1694: "f32[864]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_508);  sum_65 = squeeze_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1693, convolution_313, primals_653, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1693 = convolution_313 = primals_653 = None
    getitem_660: "f32[8, 864, 11, 11]" = convolution_backward_58[0]
    getitem_661: "f32[864, 864, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(getitem_660, relu_167, primals_652, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_660 = primals_652 = None
    getitem_663: "f32[8, 864, 11, 11]" = convolution_backward_59[0]
    getitem_664: "f32[864, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_32: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_167, 0);  relu_167 = None
    where_32: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_32, full_default, getitem_663);  le_32 = getitem_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1093: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_13, where_32);  slice_13 = where_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_66: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1093, [0, 2, 3])
    sub_329: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_312, unsqueeze_1190);  convolution_312 = unsqueeze_1190 = None
    mul_1695: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1093, sub_329)
    sum_67: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1695, [0, 2, 3]);  mul_1695 = None
    mul_1696: "f32[864]" = torch.ops.aten.mul.Tensor(sum_66, 0.0010330578512396695)
    unsqueeze_1191: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1696, 0);  mul_1696 = None
    unsqueeze_1192: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
    unsqueeze_1193: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
    mul_1697: "f32[864]" = torch.ops.aten.mul.Tensor(sum_67, 0.0010330578512396695)
    mul_1698: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_505, squeeze_505)
    mul_1699: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1697, mul_1698);  mul_1697 = mul_1698 = None
    unsqueeze_1194: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1699, 0);  mul_1699 = None
    unsqueeze_1195: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1700: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_505, primals_650);  primals_650 = None
    unsqueeze_1197: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1700, 0);  mul_1700 = None
    unsqueeze_1198: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    mul_1701: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1196);  sub_329 = unsqueeze_1196 = None
    sub_331: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1093, mul_1701);  mul_1701 = None
    sub_332: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1193);  sub_331 = None
    mul_1702: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1199);  sub_332 = unsqueeze_1199 = None
    mul_1703: "f32[864]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_505);  sum_67 = squeeze_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1702, convolution_311, primals_649, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1702 = convolution_311 = primals_649 = None
    getitem_666: "f32[8, 864, 11, 11]" = convolution_backward_60[0]
    getitem_667: "f32[864, 864, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(getitem_666, relu_166, primals_648, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_666 = primals_648 = None
    getitem_669: "f32[8, 864, 11, 11]" = convolution_backward_61[0]
    getitem_670: "f32[864, 1, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_33: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_166, 0);  relu_166 = None
    where_33: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_33, full_default, getitem_669);  le_33 = getitem_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_68: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_333: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_310, unsqueeze_1202);  convolution_310 = unsqueeze_1202 = None
    mul_1704: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_33, sub_333)
    sum_69: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1704, [0, 2, 3]);  mul_1704 = None
    mul_1705: "f32[864]" = torch.ops.aten.mul.Tensor(sum_68, 0.0010330578512396695)
    unsqueeze_1203: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1705, 0);  mul_1705 = None
    unsqueeze_1204: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1706: "f32[864]" = torch.ops.aten.mul.Tensor(sum_69, 0.0010330578512396695)
    mul_1707: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_502, squeeze_502)
    mul_1708: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1706, mul_1707);  mul_1706 = mul_1707 = None
    unsqueeze_1206: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1708, 0);  mul_1708 = None
    unsqueeze_1207: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1709: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_502, primals_646);  primals_646 = None
    unsqueeze_1209: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1709, 0);  mul_1709 = None
    unsqueeze_1210: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    mul_1710: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1208);  sub_333 = unsqueeze_1208 = None
    sub_335: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_33, mul_1710);  where_33 = mul_1710 = None
    sub_336: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1205);  sub_335 = unsqueeze_1205 = None
    mul_1711: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1211);  sub_336 = unsqueeze_1211 = None
    mul_1712: "f32[864]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_502);  sum_69 = squeeze_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1711, convolution_309, primals_645, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1711 = convolution_309 = primals_645 = None
    getitem_672: "f32[8, 864, 11, 11]" = convolution_backward_62[0]
    getitem_673: "f32[864, 864, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(getitem_672, relu_161, primals_644, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_672 = primals_644 = None
    getitem_675: "f32[8, 864, 11, 11]" = convolution_backward_63[0]
    getitem_676: "f32[864, 1, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_34: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_161, 0)
    where_34: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_34, full_default, getitem_675);  getitem_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1094: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1092, where_34);  add_1092 = where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_337: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_308, unsqueeze_1214);  convolution_308 = unsqueeze_1214 = None
    mul_1713: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1093, sub_337)
    sum_71: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1713, [0, 2, 3]);  mul_1713 = None
    mul_1715: "f32[864]" = torch.ops.aten.mul.Tensor(sum_71, 0.0010330578512396695)
    mul_1716: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_499, squeeze_499)
    mul_1717: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1715, mul_1716);  mul_1715 = mul_1716 = None
    unsqueeze_1218: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1717, 0);  mul_1717 = None
    unsqueeze_1219: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1718: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_499, primals_642);  primals_642 = None
    unsqueeze_1221: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1718, 0);  mul_1718 = None
    unsqueeze_1222: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    mul_1719: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1220);  sub_337 = unsqueeze_1220 = None
    sub_339: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1093, mul_1719);  add_1093 = mul_1719 = None
    sub_340: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1193);  sub_339 = unsqueeze_1193 = None
    mul_1720: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1223);  sub_340 = unsqueeze_1223 = None
    mul_1721: "f32[864]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_499);  sum_71 = squeeze_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1720, convolution_307, primals_641, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1720 = convolution_307 = primals_641 = None
    getitem_678: "f32[8, 864, 11, 11]" = convolution_backward_64[0]
    getitem_679: "f32[864, 864, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(getitem_678, relu_164, primals_640, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_678 = primals_640 = None
    getitem_681: "f32[8, 864, 11, 11]" = convolution_backward_65[0]
    getitem_682: "f32[864, 1, 5, 5]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_35: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_164, 0);  relu_164 = None
    where_35: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_35, full_default, getitem_681);  le_35 = getitem_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_72: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_341: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_306, unsqueeze_1226);  convolution_306 = unsqueeze_1226 = None
    mul_1722: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_35, sub_341)
    sum_73: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1722, [0, 2, 3]);  mul_1722 = None
    mul_1723: "f32[864]" = torch.ops.aten.mul.Tensor(sum_72, 0.0010330578512396695)
    unsqueeze_1227: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1723, 0);  mul_1723 = None
    unsqueeze_1228: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1724: "f32[864]" = torch.ops.aten.mul.Tensor(sum_73, 0.0010330578512396695)
    mul_1725: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_496, squeeze_496)
    mul_1726: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1724, mul_1725);  mul_1724 = mul_1725 = None
    unsqueeze_1230: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1726, 0);  mul_1726 = None
    unsqueeze_1231: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1727: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_496, primals_638);  primals_638 = None
    unsqueeze_1233: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1727, 0);  mul_1727 = None
    unsqueeze_1234: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    mul_1728: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1232);  sub_341 = unsqueeze_1232 = None
    sub_343: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_35, mul_1728);  where_35 = mul_1728 = None
    sub_344: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1229);  sub_343 = unsqueeze_1229 = None
    mul_1729: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1235);  sub_344 = unsqueeze_1235 = None
    mul_1730: "f32[864]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_496);  sum_73 = squeeze_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1729, convolution_305, primals_637, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1729 = convolution_305 = primals_637 = None
    getitem_684: "f32[8, 864, 11, 11]" = convolution_backward_66[0]
    getitem_685: "f32[864, 864, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(getitem_684, relu_161, primals_636, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_684 = primals_636 = None
    getitem_687: "f32[8, 864, 11, 11]" = convolution_backward_67[0]
    getitem_688: "f32[864, 1, 5, 5]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_36: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_34, full_default, getitem_687);  getitem_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1095: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1094, where_36);  add_1094 = where_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_7: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_12, add_859, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_399);  add_859 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1096: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1095, max_pool2d_with_indices_backward_7);  add_1095 = max_pool2d_with_indices_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_74: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_12, [0, 2, 3])
    sub_345: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_304, unsqueeze_1238);  convolution_304 = unsqueeze_1238 = None
    mul_1731: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_12, sub_345)
    sum_75: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1731, [0, 2, 3]);  mul_1731 = None
    mul_1732: "f32[864]" = torch.ops.aten.mul.Tensor(sum_74, 0.0010330578512396695)
    unsqueeze_1239: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1732, 0);  mul_1732 = None
    unsqueeze_1240: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1733: "f32[864]" = torch.ops.aten.mul.Tensor(sum_75, 0.0010330578512396695)
    mul_1734: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_493, squeeze_493)
    mul_1735: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1733, mul_1734);  mul_1733 = mul_1734 = None
    unsqueeze_1242: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1735, 0);  mul_1735 = None
    unsqueeze_1243: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1736: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_493, primals_634);  primals_634 = None
    unsqueeze_1245: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1736, 0);  mul_1736 = None
    unsqueeze_1246: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    mul_1737: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1244);  sub_345 = unsqueeze_1244 = None
    sub_347: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_12, mul_1737);  slice_12 = mul_1737 = None
    sub_348: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1241);  sub_347 = unsqueeze_1241 = None
    mul_1738: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1247);  sub_348 = unsqueeze_1247 = None
    mul_1739: "f32[864]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_493);  sum_75 = squeeze_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1738, convolution_303, primals_633, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1738 = convolution_303 = primals_633 = None
    getitem_690: "f32[8, 864, 11, 11]" = convolution_backward_68[0]
    getitem_691: "f32[864, 864, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(getitem_690, relu_162, primals_632, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_690 = primals_632 = None
    getitem_693: "f32[8, 864, 11, 11]" = convolution_backward_69[0]
    getitem_694: "f32[864, 1, 7, 7]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_37: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_162, 0);  relu_162 = None
    where_37: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_37, full_default, getitem_693);  le_37 = getitem_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_76: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_349: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_302, unsqueeze_1250);  convolution_302 = unsqueeze_1250 = None
    mul_1740: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_37, sub_349)
    sum_77: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1740, [0, 2, 3]);  mul_1740 = None
    mul_1741: "f32[864]" = torch.ops.aten.mul.Tensor(sum_76, 0.0010330578512396695)
    unsqueeze_1251: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1741, 0);  mul_1741 = None
    unsqueeze_1252: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1742: "f32[864]" = torch.ops.aten.mul.Tensor(sum_77, 0.0010330578512396695)
    mul_1743: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_490, squeeze_490)
    mul_1744: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1742, mul_1743);  mul_1742 = mul_1743 = None
    unsqueeze_1254: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1744, 0);  mul_1744 = None
    unsqueeze_1255: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1745: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_490, primals_630);  primals_630 = None
    unsqueeze_1257: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1745, 0);  mul_1745 = None
    unsqueeze_1258: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    mul_1746: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1256);  sub_349 = unsqueeze_1256 = None
    sub_351: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_37, mul_1746);  where_37 = mul_1746 = None
    sub_352: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1253);  sub_351 = unsqueeze_1253 = None
    mul_1747: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1259);  sub_352 = unsqueeze_1259 = None
    mul_1748: "f32[864]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_490);  sum_77 = squeeze_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1747, convolution_301, primals_629, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1747 = convolution_301 = primals_629 = None
    getitem_696: "f32[8, 864, 11, 11]" = convolution_backward_70[0]
    getitem_697: "f32[864, 864, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(getitem_696, relu_161, primals_628, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_696 = relu_161 = primals_628 = None
    getitem_699: "f32[8, 864, 11, 11]" = convolution_backward_71[0]
    getitem_700: "f32[864, 1, 7, 7]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_38: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_34, full_default, getitem_699);  le_34 = getitem_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1097: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1096, where_38);  add_1096 = where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_8: "f32[8, 864, 11, 11]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_11, add_854, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_393);  add_854 = getitem_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1098: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(where_30, max_pool2d_with_indices_backward_8);  where_30 = max_pool2d_with_indices_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_78: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_11, [0, 2, 3])
    sub_353: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_300, unsqueeze_1262);  convolution_300 = unsqueeze_1262 = None
    mul_1749: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_11, sub_353)
    sum_79: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1749, [0, 2, 3]);  mul_1749 = None
    mul_1750: "f32[864]" = torch.ops.aten.mul.Tensor(sum_78, 0.0010330578512396695)
    unsqueeze_1263: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1750, 0);  mul_1750 = None
    unsqueeze_1264: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1751: "f32[864]" = torch.ops.aten.mul.Tensor(sum_79, 0.0010330578512396695)
    mul_1752: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_487, squeeze_487)
    mul_1753: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1751, mul_1752);  mul_1751 = mul_1752 = None
    unsqueeze_1266: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1753, 0);  mul_1753 = None
    unsqueeze_1267: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1754: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_487, primals_626);  primals_626 = None
    unsqueeze_1269: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1754, 0);  mul_1754 = None
    unsqueeze_1270: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    mul_1755: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1268);  sub_353 = unsqueeze_1268 = None
    sub_355: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_11, mul_1755);  slice_11 = mul_1755 = None
    sub_356: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1265);  sub_355 = unsqueeze_1265 = None
    mul_1756: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1271);  sub_356 = unsqueeze_1271 = None
    mul_1757: "f32[864]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_487);  sum_79 = squeeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1756, convolution_299, primals_625, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1756 = convolution_299 = primals_625 = None
    getitem_702: "f32[8, 864, 11, 11]" = convolution_backward_72[0]
    getitem_703: "f32[864, 864, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(getitem_702, relu_160, primals_624, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_702 = primals_624 = None
    getitem_705: "f32[8, 864, 11, 11]" = convolution_backward_73[0]
    getitem_706: "f32[864, 1, 5, 5]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_39: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_160, 0);  relu_160 = None
    where_39: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_39, full_default, getitem_705);  le_39 = getitem_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_80: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_357: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_298, unsqueeze_1274);  convolution_298 = unsqueeze_1274 = None
    mul_1758: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_39, sub_357)
    sum_81: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1758, [0, 2, 3]);  mul_1758 = None
    mul_1759: "f32[864]" = torch.ops.aten.mul.Tensor(sum_80, 0.0010330578512396695)
    unsqueeze_1275: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1759, 0);  mul_1759 = None
    unsqueeze_1276: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1760: "f32[864]" = torch.ops.aten.mul.Tensor(sum_81, 0.0010330578512396695)
    mul_1761: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_484, squeeze_484)
    mul_1762: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1760, mul_1761);  mul_1760 = mul_1761 = None
    unsqueeze_1278: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1762, 0);  mul_1762 = None
    unsqueeze_1279: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1763: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_484, primals_622);  primals_622 = None
    unsqueeze_1281: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1763, 0);  mul_1763 = None
    unsqueeze_1282: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    mul_1764: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1280);  sub_357 = unsqueeze_1280 = None
    sub_359: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_39, mul_1764);  where_39 = mul_1764 = None
    sub_360: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1277);  sub_359 = unsqueeze_1277 = None
    mul_1765: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1283);  sub_360 = unsqueeze_1283 = None
    mul_1766: "f32[864]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_484);  sum_81 = squeeze_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1765, convolution_297, primals_621, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1765 = convolution_297 = primals_621 = None
    getitem_708: "f32[8, 864, 11, 11]" = convolution_backward_74[0]
    getitem_709: "f32[864, 864, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(getitem_708, relu_159, primals_620, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_708 = relu_159 = primals_620 = None
    getitem_711: "f32[8, 864, 11, 11]" = convolution_backward_75[0]
    getitem_712: "f32[864, 1, 5, 5]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_40: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_30, full_default, getitem_711);  le_30 = getitem_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1099: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(add_1098, where_40);  add_1098 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_82: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1097, [0, 2, 3])
    sub_361: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_296, unsqueeze_1286);  convolution_296 = unsqueeze_1286 = None
    mul_1767: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1097, sub_361)
    sum_83: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1767, [0, 2, 3]);  mul_1767 = None
    mul_1768: "f32[864]" = torch.ops.aten.mul.Tensor(sum_82, 0.0010330578512396695)
    unsqueeze_1287: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1768, 0);  mul_1768 = None
    unsqueeze_1288: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1769: "f32[864]" = torch.ops.aten.mul.Tensor(sum_83, 0.0010330578512396695)
    mul_1770: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_481, squeeze_481)
    mul_1771: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1769, mul_1770);  mul_1769 = mul_1770 = None
    unsqueeze_1290: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1771, 0);  mul_1771 = None
    unsqueeze_1291: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1772: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_481, primals_618);  primals_618 = None
    unsqueeze_1293: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1772, 0);  mul_1772 = None
    unsqueeze_1294: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    mul_1773: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1292);  sub_361 = unsqueeze_1292 = None
    sub_363: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1097, mul_1773);  add_1097 = mul_1773 = None
    sub_364: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1289);  sub_363 = unsqueeze_1289 = None
    mul_1774: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1295);  sub_364 = unsqueeze_1295 = None
    mul_1775: "f32[864]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_481);  sum_83 = squeeze_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1774, relu_158, primals_617, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1774 = relu_158 = primals_617 = None
    getitem_714: "f32[8, 4320, 11, 11]" = convolution_backward_76[0]
    getitem_715: "f32[864, 4320, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_41: "f32[8, 4320, 11, 11]" = torch.ops.aten.where.self(le_28, full_default, getitem_714);  le_28 = getitem_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1100: "f32[8, 4320, 11, 11]" = torch.ops.aten.add.Tensor(where_28, where_41);  where_28 = where_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    sum_84: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1099, [0, 2, 3])
    sub_365: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_1298);  cat_14 = unsqueeze_1298 = None
    mul_1776: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1099, sub_365)
    sum_85: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1776, [0, 2, 3]);  mul_1776 = None
    mul_1777: "f32[864]" = torch.ops.aten.mul.Tensor(sum_84, 0.0010330578512396695)
    unsqueeze_1299: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1777, 0);  mul_1777 = None
    unsqueeze_1300: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1778: "f32[864]" = torch.ops.aten.mul.Tensor(sum_85, 0.0010330578512396695)
    mul_1779: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_478, squeeze_478)
    mul_1780: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1778, mul_1779);  mul_1778 = mul_1779 = None
    unsqueeze_1302: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1780, 0);  mul_1780 = None
    unsqueeze_1303: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1781: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_478, primals_615);  primals_615 = None
    unsqueeze_1305: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1781, 0);  mul_1781 = None
    unsqueeze_1306: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    mul_1782: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1304);  sub_365 = unsqueeze_1304 = None
    sub_367: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1099, mul_1782);  add_1099 = mul_1782 = None
    sub_368: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1301);  sub_367 = unsqueeze_1301 = None
    mul_1783: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1307);  sub_368 = unsqueeze_1307 = None
    mul_1784: "f32[864]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_478);  sum_85 = squeeze_478 = None
    slice_16: "f32[8, 432, 11, 11]" = torch.ops.aten.slice.Tensor(mul_1783, 1, 0, 432)
    slice_17: "f32[8, 432, 11, 11]" = torch.ops.aten.slice.Tensor(mul_1783, 1, 432, 864);  mul_1783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(slice_17, avg_pool2d_7, primals_614, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_17 = avg_pool2d_7 = primals_614 = None
    getitem_717: "f32[8, 2160, 11, 11]" = convolution_backward_77[0]
    getitem_718: "f32[432, 2160, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    avg_pool2d_backward: "f32[8, 2160, 21, 21]" = torch.ops.aten.avg_pool2d_backward.default(getitem_717, constant_pad_nd_39, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_717 = constant_pad_nd_39 = None
    constant_pad_nd_40: "f32[8, 2160, 21, 21]" = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward, [1, -1, 1, -1]);  avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(slice_16, avg_pool2d_6, primals_613, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_16 = avg_pool2d_6 = primals_613 = None
    getitem_720: "f32[8, 2160, 11, 11]" = convolution_backward_78[0]
    getitem_721: "f32[432, 2160, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    avg_pool2d_backward_1: "f32[8, 2160, 21, 21]" = torch.ops.aten.avg_pool2d_backward.default(getitem_720, relu_143, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    add_1101: "f32[8, 2160, 21, 21]" = torch.ops.aten.add.Tensor(constant_pad_nd_40, avg_pool2d_backward_1);  constant_pad_nd_40 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    le_42: "b8[8, 2160, 21, 21]" = torch.ops.aten.le.Scalar(relu_143, 0)
    where_42: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_42, full_default, add_1101);  add_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_18: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1100, 1, 0, 864)
    slice_19: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1100, 1, 864, 1728)
    slice_20: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1100, 1, 1728, 2592)
    slice_21: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1100, 1, 2592, 3456)
    slice_22: "f32[8, 864, 11, 11]" = torch.ops.aten.slice.Tensor(add_1100, 1, 3456, 4320);  add_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_86: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_22, [0, 2, 3])
    sub_369: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_293, unsqueeze_1310);  convolution_293 = unsqueeze_1310 = None
    mul_1785: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_22, sub_369)
    sum_87: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1785, [0, 2, 3]);  mul_1785 = None
    mul_1786: "f32[864]" = torch.ops.aten.mul.Tensor(sum_86, 0.0010330578512396695)
    unsqueeze_1311: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1786, 0);  mul_1786 = None
    unsqueeze_1312: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1787: "f32[864]" = torch.ops.aten.mul.Tensor(sum_87, 0.0010330578512396695)
    mul_1788: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_475, squeeze_475)
    mul_1789: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1787, mul_1788);  mul_1787 = mul_1788 = None
    unsqueeze_1314: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1789, 0);  mul_1789 = None
    unsqueeze_1315: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1790: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_475, primals_611);  primals_611 = None
    unsqueeze_1317: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1790, 0);  mul_1790 = None
    unsqueeze_1318: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    mul_1791: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1316);  sub_369 = unsqueeze_1316 = None
    sub_371: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_22, mul_1791);  mul_1791 = None
    sub_372: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1313);  sub_371 = None
    mul_1792: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1319);  sub_372 = unsqueeze_1319 = None
    mul_1793: "f32[864]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_475);  sum_87 = squeeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1792, constant_pad_nd_38, primals_26, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1792 = constant_pad_nd_38 = primals_26 = None
    getitem_723: "f32[8, 864, 21, 21]" = convolution_backward_79[0]
    getitem_724: "f32[864, 864, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_43: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_43, full_default, getitem_723);  getitem_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_373: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_292, unsqueeze_1322);  convolution_292 = unsqueeze_1322 = None
    mul_1794: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_22, sub_373)
    sum_89: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1794, [0, 2, 3]);  mul_1794 = None
    mul_1796: "f32[864]" = torch.ops.aten.mul.Tensor(sum_89, 0.0010330578512396695)
    mul_1797: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_472, squeeze_472)
    mul_1798: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1796, mul_1797);  mul_1796 = mul_1797 = None
    unsqueeze_1326: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1798, 0);  mul_1798 = None
    unsqueeze_1327: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1799: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_472, primals_609);  primals_609 = None
    unsqueeze_1329: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1799, 0);  mul_1799 = None
    unsqueeze_1330: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    mul_1800: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1328);  sub_373 = unsqueeze_1328 = None
    sub_375: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_22, mul_1800);  slice_22 = mul_1800 = None
    sub_376: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1313);  sub_375 = unsqueeze_1313 = None
    mul_1801: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1331);  sub_376 = unsqueeze_1331 = None
    mul_1802: "f32[864]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_472);  sum_89 = squeeze_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1801, convolution_291, primals_608, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1801 = convolution_291 = primals_608 = None
    getitem_726: "f32[8, 864, 11, 11]" = convolution_backward_80[0]
    getitem_727: "f32[864, 864, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(getitem_726, relu_155, primals_607, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_726 = primals_607 = None
    getitem_729: "f32[8, 864, 11, 11]" = convolution_backward_81[0]
    getitem_730: "f32[864, 1, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_44: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_155, 0);  relu_155 = None
    where_44: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_44, full_default, getitem_729);  le_44 = getitem_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_90: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_377: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_290, unsqueeze_1334);  convolution_290 = unsqueeze_1334 = None
    mul_1803: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_44, sub_377)
    sum_91: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1803, [0, 2, 3]);  mul_1803 = None
    mul_1804: "f32[864]" = torch.ops.aten.mul.Tensor(sum_90, 0.0010330578512396695)
    unsqueeze_1335: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1804, 0);  mul_1804 = None
    unsqueeze_1336: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1805: "f32[864]" = torch.ops.aten.mul.Tensor(sum_91, 0.0010330578512396695)
    mul_1806: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_469, squeeze_469)
    mul_1807: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1805, mul_1806);  mul_1805 = mul_1806 = None
    unsqueeze_1338: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1807, 0);  mul_1807 = None
    unsqueeze_1339: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1808: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_469, primals_605);  primals_605 = None
    unsqueeze_1341: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1808, 0);  mul_1808 = None
    unsqueeze_1342: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    mul_1809: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1340);  sub_377 = unsqueeze_1340 = None
    sub_379: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_44, mul_1809);  where_44 = mul_1809 = None
    sub_380: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1337);  sub_379 = unsqueeze_1337 = None
    mul_1810: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1343);  sub_380 = unsqueeze_1343 = None
    mul_1811: "f32[864]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_469);  sum_91 = squeeze_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1810, convolution_289, primals_604, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1810 = convolution_289 = primals_604 = None
    getitem_732: "f32[8, 864, 11, 11]" = convolution_backward_82[0]
    getitem_733: "f32[864, 864, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(getitem_732, constant_pad_nd_37, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_732 = constant_pad_nd_37 = primals_25 = None
    getitem_735: "f32[8, 864, 23, 23]" = convolution_backward_83[0]
    getitem_736: "f32[864, 1, 3, 3]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_42: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_735, [-1, -1, -1, -1]);  getitem_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_45: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_45, full_default, constant_pad_nd_42);  constant_pad_nd_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_9: "f32[8, 864, 23, 23]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_21, constant_pad_nd_33, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_363)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_43: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_9, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1102: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(where_43, constant_pad_nd_43);  where_43 = constant_pad_nd_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_92: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_21, [0, 2, 3])
    sub_381: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_288, unsqueeze_1346);  convolution_288 = unsqueeze_1346 = None
    mul_1812: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_21, sub_381)
    sum_93: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1812, [0, 2, 3]);  mul_1812 = None
    mul_1813: "f32[864]" = torch.ops.aten.mul.Tensor(sum_92, 0.0010330578512396695)
    unsqueeze_1347: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1813, 0);  mul_1813 = None
    unsqueeze_1348: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1814: "f32[864]" = torch.ops.aten.mul.Tensor(sum_93, 0.0010330578512396695)
    mul_1815: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_466, squeeze_466)
    mul_1816: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1814, mul_1815);  mul_1814 = mul_1815 = None
    unsqueeze_1350: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1816, 0);  mul_1816 = None
    unsqueeze_1351: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1817: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_466, primals_602);  primals_602 = None
    unsqueeze_1353: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1817, 0);  mul_1817 = None
    unsqueeze_1354: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    mul_1818: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1352);  sub_381 = unsqueeze_1352 = None
    sub_383: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_21, mul_1818);  slice_21 = mul_1818 = None
    sub_384: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1349);  sub_383 = unsqueeze_1349 = None
    mul_1819: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1355);  sub_384 = unsqueeze_1355 = None
    mul_1820: "f32[864]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_466);  sum_93 = squeeze_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1819, convolution_287, primals_601, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1819 = convolution_287 = primals_601 = None
    getitem_738: "f32[8, 864, 11, 11]" = convolution_backward_84[0]
    getitem_739: "f32[864, 864, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(getitem_738, relu_153, primals_600, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_738 = primals_600 = None
    getitem_741: "f32[8, 864, 11, 11]" = convolution_backward_85[0]
    getitem_742: "f32[864, 1, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_46: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_153, 0);  relu_153 = None
    where_46: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_46, full_default, getitem_741);  le_46 = getitem_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_94: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_385: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_286, unsqueeze_1358);  convolution_286 = unsqueeze_1358 = None
    mul_1821: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_46, sub_385)
    sum_95: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1821, [0, 2, 3]);  mul_1821 = None
    mul_1822: "f32[864]" = torch.ops.aten.mul.Tensor(sum_94, 0.0010330578512396695)
    unsqueeze_1359: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1822, 0);  mul_1822 = None
    unsqueeze_1360: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1823: "f32[864]" = torch.ops.aten.mul.Tensor(sum_95, 0.0010330578512396695)
    mul_1824: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_463, squeeze_463)
    mul_1825: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1823, mul_1824);  mul_1823 = mul_1824 = None
    unsqueeze_1362: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1825, 0);  mul_1825 = None
    unsqueeze_1363: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1826: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_463, primals_598);  primals_598 = None
    unsqueeze_1365: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1826, 0);  mul_1826 = None
    unsqueeze_1366: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    mul_1827: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1364);  sub_385 = unsqueeze_1364 = None
    sub_387: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_46, mul_1827);  where_46 = mul_1827 = None
    sub_388: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1361);  sub_387 = unsqueeze_1361 = None
    mul_1828: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1367);  sub_388 = unsqueeze_1367 = None
    mul_1829: "f32[864]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_463);  sum_95 = squeeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1828, convolution_285, primals_597, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1828 = convolution_285 = primals_597 = None
    getitem_744: "f32[8, 864, 11, 11]" = convolution_backward_86[0]
    getitem_745: "f32[864, 864, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(getitem_744, relu_152, primals_596, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_744 = primals_596 = None
    getitem_747: "f32[8, 864, 11, 11]" = convolution_backward_87[0]
    getitem_748: "f32[864, 1, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_47: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_152, 0);  relu_152 = None
    where_47: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_47, full_default, getitem_747);  le_47 = getitem_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1103: "f32[8, 864, 11, 11]" = torch.ops.aten.add.Tensor(slice_20, where_47);  slice_20 = where_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_96: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1103, [0, 2, 3])
    sub_389: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_284, unsqueeze_1370);  convolution_284 = unsqueeze_1370 = None
    mul_1830: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1103, sub_389)
    sum_97: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1830, [0, 2, 3]);  mul_1830 = None
    mul_1831: "f32[864]" = torch.ops.aten.mul.Tensor(sum_96, 0.0010330578512396695)
    unsqueeze_1371: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1831, 0);  mul_1831 = None
    unsqueeze_1372: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1832: "f32[864]" = torch.ops.aten.mul.Tensor(sum_97, 0.0010330578512396695)
    mul_1833: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_460, squeeze_460)
    mul_1834: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1832, mul_1833);  mul_1832 = mul_1833 = None
    unsqueeze_1374: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1834, 0);  mul_1834 = None
    unsqueeze_1375: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1835: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_460, primals_594);  primals_594 = None
    unsqueeze_1377: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1835, 0);  mul_1835 = None
    unsqueeze_1378: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    mul_1836: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1376);  sub_389 = unsqueeze_1376 = None
    sub_391: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1103, mul_1836);  mul_1836 = None
    sub_392: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1373);  sub_391 = None
    mul_1837: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1379);  sub_392 = unsqueeze_1379 = None
    mul_1838: "f32[864]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_460);  sum_97 = squeeze_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1837, convolution_283, primals_593, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1837 = convolution_283 = primals_593 = None
    getitem_750: "f32[8, 864, 11, 11]" = convolution_backward_88[0]
    getitem_751: "f32[864, 864, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(getitem_750, relu_151, primals_592, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_750 = primals_592 = None
    getitem_753: "f32[8, 864, 11, 11]" = convolution_backward_89[0]
    getitem_754: "f32[864, 1, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_48: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_151, 0);  relu_151 = None
    where_48: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_48, full_default, getitem_753);  le_48 = getitem_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_98: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_393: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_282, unsqueeze_1382);  convolution_282 = unsqueeze_1382 = None
    mul_1839: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_48, sub_393)
    sum_99: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1839, [0, 2, 3]);  mul_1839 = None
    mul_1840: "f32[864]" = torch.ops.aten.mul.Tensor(sum_98, 0.0010330578512396695)
    unsqueeze_1383: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1840, 0);  mul_1840 = None
    unsqueeze_1384: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1841: "f32[864]" = torch.ops.aten.mul.Tensor(sum_99, 0.0010330578512396695)
    mul_1842: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_457, squeeze_457)
    mul_1843: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1841, mul_1842);  mul_1841 = mul_1842 = None
    unsqueeze_1386: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1843, 0);  mul_1843 = None
    unsqueeze_1387: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1844: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_457, primals_590);  primals_590 = None
    unsqueeze_1389: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1844, 0);  mul_1844 = None
    unsqueeze_1390: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    mul_1845: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1388);  sub_393 = unsqueeze_1388 = None
    sub_395: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_48, mul_1845);  where_48 = mul_1845 = None
    sub_396: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1385);  sub_395 = unsqueeze_1385 = None
    mul_1846: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1391);  sub_396 = unsqueeze_1391 = None
    mul_1847: "f32[864]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_457);  sum_99 = squeeze_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1846, convolution_281, primals_589, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1846 = convolution_281 = primals_589 = None
    getitem_756: "f32[8, 864, 11, 11]" = convolution_backward_90[0]
    getitem_757: "f32[864, 864, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(getitem_756, constant_pad_nd_35, primals_24, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_756 = constant_pad_nd_35 = primals_24 = None
    getitem_759: "f32[8, 864, 23, 23]" = convolution_backward_91[0]
    getitem_760: "f32[864, 1, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_44: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_759, [-1, -1, -1, -1]);  getitem_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_49: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_43, full_default, constant_pad_nd_44);  constant_pad_nd_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1104: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(add_1102, where_49);  add_1102 = where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_397: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_280, unsqueeze_1394);  convolution_280 = unsqueeze_1394 = None
    mul_1848: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(add_1103, sub_397)
    sum_101: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1848, [0, 2, 3]);  mul_1848 = None
    mul_1850: "f32[864]" = torch.ops.aten.mul.Tensor(sum_101, 0.0010330578512396695)
    mul_1851: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_454, squeeze_454)
    mul_1852: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1850, mul_1851);  mul_1850 = mul_1851 = None
    unsqueeze_1398: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1852, 0);  mul_1852 = None
    unsqueeze_1399: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
    unsqueeze_1400: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
    mul_1853: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_454, primals_587);  primals_587 = None
    unsqueeze_1401: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1853, 0);  mul_1853 = None
    unsqueeze_1402: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
    unsqueeze_1403: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
    mul_1854: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1400);  sub_397 = unsqueeze_1400 = None
    sub_399: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(add_1103, mul_1854);  add_1103 = mul_1854 = None
    sub_400: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1373);  sub_399 = unsqueeze_1373 = None
    mul_1855: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1403);  sub_400 = unsqueeze_1403 = None
    mul_1856: "f32[864]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_454);  sum_101 = squeeze_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1855, convolution_279, primals_586, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1855 = convolution_279 = primals_586 = None
    getitem_762: "f32[8, 864, 11, 11]" = convolution_backward_92[0]
    getitem_763: "f32[864, 864, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(getitem_762, relu_149, primals_585, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_762 = primals_585 = None
    getitem_765: "f32[8, 864, 11, 11]" = convolution_backward_93[0]
    getitem_766: "f32[864, 1, 5, 5]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_50: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_149, 0);  relu_149 = None
    where_50: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_50, full_default, getitem_765);  le_50 = getitem_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_102: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_401: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_278, unsqueeze_1406);  convolution_278 = unsqueeze_1406 = None
    mul_1857: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_50, sub_401)
    sum_103: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1857, [0, 2, 3]);  mul_1857 = None
    mul_1858: "f32[864]" = torch.ops.aten.mul.Tensor(sum_102, 0.0010330578512396695)
    unsqueeze_1407: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1858, 0);  mul_1858 = None
    unsqueeze_1408: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 2);  unsqueeze_1407 = None
    unsqueeze_1409: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 3);  unsqueeze_1408 = None
    mul_1859: "f32[864]" = torch.ops.aten.mul.Tensor(sum_103, 0.0010330578512396695)
    mul_1860: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_451, squeeze_451)
    mul_1861: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1859, mul_1860);  mul_1859 = mul_1860 = None
    unsqueeze_1410: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1861, 0);  mul_1861 = None
    unsqueeze_1411: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
    unsqueeze_1412: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
    mul_1862: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_451, primals_583);  primals_583 = None
    unsqueeze_1413: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1862, 0);  mul_1862 = None
    unsqueeze_1414: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
    unsqueeze_1415: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
    mul_1863: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1412);  sub_401 = unsqueeze_1412 = None
    sub_403: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_50, mul_1863);  where_50 = mul_1863 = None
    sub_404: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1409);  sub_403 = unsqueeze_1409 = None
    mul_1864: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1415);  sub_404 = unsqueeze_1415 = None
    mul_1865: "f32[864]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_451);  sum_103 = squeeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1864, convolution_277, primals_582, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1864 = convolution_277 = primals_582 = None
    getitem_768: "f32[8, 864, 11, 11]" = convolution_backward_94[0]
    getitem_769: "f32[864, 864, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(getitem_768, constant_pad_nd_34, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_768 = constant_pad_nd_34 = primals_23 = None
    getitem_771: "f32[8, 864, 25, 25]" = convolution_backward_95[0]
    getitem_772: "f32[864, 1, 5, 5]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_45: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_771, [-2, -2, -2, -2]);  getitem_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_51: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_43, full_default, constant_pad_nd_45);  constant_pad_nd_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1105: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(add_1104, where_51);  add_1104 = where_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_10: "f32[8, 864, 23, 23]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_19, constant_pad_nd_33, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_363);  constant_pad_nd_33 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_46: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_10, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1106: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(add_1105, constant_pad_nd_46);  add_1105 = constant_pad_nd_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_104: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_19, [0, 2, 3])
    sub_405: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_276, unsqueeze_1418);  convolution_276 = unsqueeze_1418 = None
    mul_1866: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_19, sub_405)
    sum_105: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1866, [0, 2, 3]);  mul_1866 = None
    mul_1867: "f32[864]" = torch.ops.aten.mul.Tensor(sum_104, 0.0010330578512396695)
    unsqueeze_1419: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1867, 0);  mul_1867 = None
    unsqueeze_1420: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 2);  unsqueeze_1419 = None
    unsqueeze_1421: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 3);  unsqueeze_1420 = None
    mul_1868: "f32[864]" = torch.ops.aten.mul.Tensor(sum_105, 0.0010330578512396695)
    mul_1869: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_448, squeeze_448)
    mul_1870: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1868, mul_1869);  mul_1868 = mul_1869 = None
    unsqueeze_1422: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1870, 0);  mul_1870 = None
    unsqueeze_1423: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
    unsqueeze_1424: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
    mul_1871: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_448, primals_580);  primals_580 = None
    unsqueeze_1425: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1871, 0);  mul_1871 = None
    unsqueeze_1426: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
    unsqueeze_1427: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
    mul_1872: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1424);  sub_405 = unsqueeze_1424 = None
    sub_407: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_19, mul_1872);  slice_19 = mul_1872 = None
    sub_408: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1421);  sub_407 = unsqueeze_1421 = None
    mul_1873: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1427);  sub_408 = unsqueeze_1427 = None
    mul_1874: "f32[864]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_448);  sum_105 = squeeze_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1873, convolution_275, primals_579, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1873 = convolution_275 = primals_579 = None
    getitem_774: "f32[8, 864, 11, 11]" = convolution_backward_96[0]
    getitem_775: "f32[864, 864, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(getitem_774, relu_147, primals_578, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_774 = primals_578 = None
    getitem_777: "f32[8, 864, 11, 11]" = convolution_backward_97[0]
    getitem_778: "f32[864, 1, 7, 7]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_52: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_147, 0);  relu_147 = None
    where_52: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_52, full_default, getitem_777);  le_52 = getitem_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_106: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_409: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_274, unsqueeze_1430);  convolution_274 = unsqueeze_1430 = None
    mul_1875: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_52, sub_409)
    sum_107: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1875, [0, 2, 3]);  mul_1875 = None
    mul_1876: "f32[864]" = torch.ops.aten.mul.Tensor(sum_106, 0.0010330578512396695)
    unsqueeze_1431: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1876, 0);  mul_1876 = None
    unsqueeze_1432: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
    unsqueeze_1433: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
    mul_1877: "f32[864]" = torch.ops.aten.mul.Tensor(sum_107, 0.0010330578512396695)
    mul_1878: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_445, squeeze_445)
    mul_1879: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1877, mul_1878);  mul_1877 = mul_1878 = None
    unsqueeze_1434: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1879, 0);  mul_1879 = None
    unsqueeze_1435: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
    unsqueeze_1436: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
    mul_1880: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_445, primals_576);  primals_576 = None
    unsqueeze_1437: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1880, 0);  mul_1880 = None
    unsqueeze_1438: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
    unsqueeze_1439: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
    mul_1881: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1436);  sub_409 = unsqueeze_1436 = None
    sub_411: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_52, mul_1881);  where_52 = mul_1881 = None
    sub_412: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1433);  sub_411 = unsqueeze_1433 = None
    mul_1882: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1439);  sub_412 = unsqueeze_1439 = None
    mul_1883: "f32[864]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_445);  sum_107 = squeeze_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1882, convolution_273, primals_575, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1882 = convolution_273 = primals_575 = None
    getitem_780: "f32[8, 864, 11, 11]" = convolution_backward_98[0]
    getitem_781: "f32[864, 864, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(getitem_780, constant_pad_nd_32, primals_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_780 = constant_pad_nd_32 = primals_22 = None
    getitem_783: "f32[8, 864, 27, 27]" = convolution_backward_99[0]
    getitem_784: "f32[864, 1, 7, 7]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_47: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_783, [-3, -3, -3, -3]);  getitem_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_53: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_43, full_default, constant_pad_nd_47);  le_43 = constant_pad_nd_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1107: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(add_1106, where_53);  add_1106 = where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_11: "f32[8, 864, 23, 23]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_18, constant_pad_nd_31, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_357);  constant_pad_nd_31 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_48: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_11, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1108: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(where_45, constant_pad_nd_48);  where_45 = constant_pad_nd_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_108: "f32[864]" = torch.ops.aten.sum.dim_IntList(slice_18, [0, 2, 3])
    sub_413: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_272, unsqueeze_1442);  convolution_272 = unsqueeze_1442 = None
    mul_1884: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(slice_18, sub_413)
    sum_109: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1884, [0, 2, 3]);  mul_1884 = None
    mul_1885: "f32[864]" = torch.ops.aten.mul.Tensor(sum_108, 0.0010330578512396695)
    unsqueeze_1443: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1885, 0);  mul_1885 = None
    unsqueeze_1444: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 2);  unsqueeze_1443 = None
    unsqueeze_1445: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 3);  unsqueeze_1444 = None
    mul_1886: "f32[864]" = torch.ops.aten.mul.Tensor(sum_109, 0.0010330578512396695)
    mul_1887: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_442, squeeze_442)
    mul_1888: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1886, mul_1887);  mul_1886 = mul_1887 = None
    unsqueeze_1446: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1888, 0);  mul_1888 = None
    unsqueeze_1447: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
    unsqueeze_1448: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
    mul_1889: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_442, primals_573);  primals_573 = None
    unsqueeze_1449: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1889, 0);  mul_1889 = None
    unsqueeze_1450: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
    unsqueeze_1451: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
    mul_1890: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1448);  sub_413 = unsqueeze_1448 = None
    sub_415: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(slice_18, mul_1890);  slice_18 = mul_1890 = None
    sub_416: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1445);  sub_415 = unsqueeze_1445 = None
    mul_1891: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1451);  sub_416 = unsqueeze_1451 = None
    mul_1892: "f32[864]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_442);  sum_109 = squeeze_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1891, convolution_271, primals_572, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1891 = convolution_271 = primals_572 = None
    getitem_786: "f32[8, 864, 11, 11]" = convolution_backward_100[0]
    getitem_787: "f32[864, 864, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(getitem_786, relu_145, primals_571, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_786 = primals_571 = None
    getitem_789: "f32[8, 864, 11, 11]" = convolution_backward_101[0]
    getitem_790: "f32[864, 1, 5, 5]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_54: "b8[8, 864, 11, 11]" = torch.ops.aten.le.Scalar(relu_145, 0);  relu_145 = None
    where_54: "f32[8, 864, 11, 11]" = torch.ops.aten.where.self(le_54, full_default, getitem_789);  le_54 = getitem_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_110: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_417: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(convolution_270, unsqueeze_1454);  convolution_270 = unsqueeze_1454 = None
    mul_1893: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(where_54, sub_417)
    sum_111: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1893, [0, 2, 3]);  mul_1893 = None
    mul_1894: "f32[864]" = torch.ops.aten.mul.Tensor(sum_110, 0.0010330578512396695)
    unsqueeze_1455: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1894, 0);  mul_1894 = None
    unsqueeze_1456: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 2);  unsqueeze_1455 = None
    unsqueeze_1457: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 3);  unsqueeze_1456 = None
    mul_1895: "f32[864]" = torch.ops.aten.mul.Tensor(sum_111, 0.0010330578512396695)
    mul_1896: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_439, squeeze_439)
    mul_1897: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1895, mul_1896);  mul_1895 = mul_1896 = None
    unsqueeze_1458: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1897, 0);  mul_1897 = None
    unsqueeze_1459: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
    unsqueeze_1460: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
    mul_1898: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_439, primals_569);  primals_569 = None
    unsqueeze_1461: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1898, 0);  mul_1898 = None
    unsqueeze_1462: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
    unsqueeze_1463: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
    mul_1899: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1460);  sub_417 = unsqueeze_1460 = None
    sub_419: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(where_54, mul_1899);  where_54 = mul_1899 = None
    sub_420: "f32[8, 864, 11, 11]" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1457);  sub_419 = unsqueeze_1457 = None
    mul_1900: "f32[8, 864, 11, 11]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1463);  sub_420 = unsqueeze_1463 = None
    mul_1901: "f32[864]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_439);  sum_111 = squeeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1900, convolution_269, primals_568, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1900 = convolution_269 = primals_568 = None
    getitem_792: "f32[8, 864, 11, 11]" = convolution_backward_102[0]
    getitem_793: "f32[864, 864, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(getitem_792, constant_pad_nd_30, primals_21, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 864, [True, True, False]);  getitem_792 = constant_pad_nd_30 = primals_21 = None
    getitem_795: "f32[8, 864, 25, 25]" = convolution_backward_103[0]
    getitem_796: "f32[864, 1, 5, 5]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_49: "f32[8, 864, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_795, [-2, -2, -2, -2]);  getitem_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_55: "f32[8, 864, 21, 21]" = torch.ops.aten.where.self(le_45, full_default, constant_pad_nd_49);  le_45 = constant_pad_nd_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1109: "f32[8, 864, 21, 21]" = torch.ops.aten.add.Tensor(add_1108, where_55);  add_1108 = where_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_112: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1107, [0, 2, 3])
    sub_421: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_268, unsqueeze_1466);  convolution_268 = unsqueeze_1466 = None
    mul_1902: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(add_1107, sub_421)
    sum_113: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1902, [0, 2, 3]);  mul_1902 = None
    mul_1903: "f32[864]" = torch.ops.aten.mul.Tensor(sum_112, 0.0002834467120181406)
    unsqueeze_1467: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1903, 0);  mul_1903 = None
    unsqueeze_1468: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 2);  unsqueeze_1467 = None
    unsqueeze_1469: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 3);  unsqueeze_1468 = None
    mul_1904: "f32[864]" = torch.ops.aten.mul.Tensor(sum_113, 0.0002834467120181406)
    mul_1905: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_436, squeeze_436)
    mul_1906: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1904, mul_1905);  mul_1904 = mul_1905 = None
    unsqueeze_1470: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1906, 0);  mul_1906 = None
    unsqueeze_1471: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
    unsqueeze_1472: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
    mul_1907: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_436, primals_566);  primals_566 = None
    unsqueeze_1473: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1907, 0);  mul_1907 = None
    unsqueeze_1474: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
    unsqueeze_1475: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
    mul_1908: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1472);  sub_421 = unsqueeze_1472 = None
    sub_423: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(add_1107, mul_1908);  add_1107 = mul_1908 = None
    sub_424: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1469);  sub_423 = unsqueeze_1469 = None
    mul_1909: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1475);  sub_424 = unsqueeze_1475 = None
    mul_1910: "f32[864]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_436);  sum_113 = squeeze_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1909, relu_143, primals_565, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1909 = relu_143 = primals_565 = None
    getitem_798: "f32[8, 2160, 21, 21]" = convolution_backward_104[0]
    getitem_799: "f32[864, 2160, 1, 1]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_56: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_42, full_default, getitem_798);  le_42 = getitem_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1110: "f32[8, 2160, 21, 21]" = torch.ops.aten.add.Tensor(where_42, where_56);  where_42 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_114: "f32[864]" = torch.ops.aten.sum.dim_IntList(add_1109, [0, 2, 3])
    sub_425: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_267, unsqueeze_1478);  convolution_267 = unsqueeze_1478 = None
    mul_1911: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(add_1109, sub_425)
    sum_115: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_1911, [0, 2, 3]);  mul_1911 = None
    mul_1912: "f32[864]" = torch.ops.aten.mul.Tensor(sum_114, 0.0002834467120181406)
    unsqueeze_1479: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1912, 0);  mul_1912 = None
    unsqueeze_1480: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 2);  unsqueeze_1479 = None
    unsqueeze_1481: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 3);  unsqueeze_1480 = None
    mul_1913: "f32[864]" = torch.ops.aten.mul.Tensor(sum_115, 0.0002834467120181406)
    mul_1914: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_433, squeeze_433)
    mul_1915: "f32[864]" = torch.ops.aten.mul.Tensor(mul_1913, mul_1914);  mul_1913 = mul_1914 = None
    unsqueeze_1482: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1915, 0);  mul_1915 = None
    unsqueeze_1483: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
    unsqueeze_1484: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
    mul_1916: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_433, primals_563);  primals_563 = None
    unsqueeze_1485: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_1916, 0);  mul_1916 = None
    unsqueeze_1486: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
    unsqueeze_1487: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
    mul_1917: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1484);  sub_425 = unsqueeze_1484 = None
    sub_427: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(add_1109, mul_1917);  add_1109 = mul_1917 = None
    sub_428: "f32[8, 864, 21, 21]" = torch.ops.aten.sub.Tensor(sub_427, unsqueeze_1481);  sub_427 = unsqueeze_1481 = None
    mul_1918: "f32[8, 864, 21, 21]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1487);  sub_428 = unsqueeze_1487 = None
    mul_1919: "f32[864]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_433);  sum_115 = squeeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1918, relu_129, primals_562, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1918 = primals_562 = None
    getitem_801: "f32[8, 2160, 21, 21]" = convolution_backward_105[0]
    getitem_802: "f32[864, 2160, 1, 1]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_57: "b8[8, 2160, 21, 21]" = torch.ops.aten.le.Scalar(relu_129, 0)
    where_57: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_57, full_default, getitem_801);  getitem_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_23: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1110, 1, 0, 432)
    slice_24: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1110, 1, 432, 864)
    slice_25: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1110, 1, 864, 1296)
    slice_26: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1110, 1, 1296, 1728)
    slice_27: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1110, 1, 1728, 2160);  add_1110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_116: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_27, [0, 2, 3])
    sub_429: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_266, unsqueeze_1490);  convolution_266 = unsqueeze_1490 = None
    mul_1920: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_27, sub_429)
    sum_117: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1920, [0, 2, 3]);  mul_1920 = None
    mul_1921: "f32[432]" = torch.ops.aten.mul.Tensor(sum_116, 0.0002834467120181406)
    unsqueeze_1491: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1921, 0);  mul_1921 = None
    unsqueeze_1492: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 2);  unsqueeze_1491 = None
    unsqueeze_1493: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 3);  unsqueeze_1492 = None
    mul_1922: "f32[432]" = torch.ops.aten.mul.Tensor(sum_117, 0.0002834467120181406)
    mul_1923: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_430, squeeze_430)
    mul_1924: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1922, mul_1923);  mul_1922 = mul_1923 = None
    unsqueeze_1494: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1924, 0);  mul_1924 = None
    unsqueeze_1495: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
    unsqueeze_1496: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
    mul_1925: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_430, primals_560);  primals_560 = None
    unsqueeze_1497: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1925, 0);  mul_1925 = None
    unsqueeze_1498: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
    unsqueeze_1499: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
    mul_1926: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1496);  sub_429 = unsqueeze_1496 = None
    sub_431: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_27, mul_1926);  mul_1926 = None
    sub_432: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_431, unsqueeze_1493);  sub_431 = unsqueeze_1493 = None
    mul_1927: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1499);  sub_432 = unsqueeze_1499 = None
    mul_1928: "f32[432]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_430);  sum_117 = squeeze_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_1927, convolution_265, primals_559, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1927 = convolution_265 = primals_559 = None
    getitem_804: "f32[8, 432, 21, 21]" = convolution_backward_106[0]
    getitem_805: "f32[432, 432, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(getitem_804, relu_141, primals_558, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_804 = primals_558 = None
    getitem_807: "f32[8, 432, 21, 21]" = convolution_backward_107[0]
    getitem_808: "f32[432, 1, 3, 3]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_58: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_141, 0);  relu_141 = None
    where_58: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_58, full_default, getitem_807);  le_58 = getitem_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_118: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_433: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_264, unsqueeze_1502);  convolution_264 = unsqueeze_1502 = None
    mul_1929: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_58, sub_433)
    sum_119: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1929, [0, 2, 3]);  mul_1929 = None
    mul_1930: "f32[432]" = torch.ops.aten.mul.Tensor(sum_118, 0.0002834467120181406)
    unsqueeze_1503: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1930, 0);  mul_1930 = None
    unsqueeze_1504: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
    unsqueeze_1505: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
    mul_1931: "f32[432]" = torch.ops.aten.mul.Tensor(sum_119, 0.0002834467120181406)
    mul_1932: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_427, squeeze_427)
    mul_1933: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1931, mul_1932);  mul_1931 = mul_1932 = None
    unsqueeze_1506: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1933, 0);  mul_1933 = None
    unsqueeze_1507: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
    unsqueeze_1508: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
    mul_1934: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_427, primals_556);  primals_556 = None
    unsqueeze_1509: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1934, 0);  mul_1934 = None
    unsqueeze_1510: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
    unsqueeze_1511: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
    mul_1935: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1508);  sub_433 = unsqueeze_1508 = None
    sub_435: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_58, mul_1935);  where_58 = mul_1935 = None
    sub_436: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_435, unsqueeze_1505);  sub_435 = unsqueeze_1505 = None
    mul_1936: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1511);  sub_436 = unsqueeze_1511 = None
    mul_1937: "f32[432]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_427);  sum_119 = squeeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1936, convolution_263, primals_555, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1936 = convolution_263 = primals_555 = None
    getitem_810: "f32[8, 432, 21, 21]" = convolution_backward_108[0]
    getitem_811: "f32[432, 432, 1, 1]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(getitem_810, relu_130, primals_554, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_810 = primals_554 = None
    getitem_813: "f32[8, 432, 21, 21]" = convolution_backward_109[0]
    getitem_814: "f32[432, 1, 3, 3]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_59: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_130, 0)
    where_59: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_59, full_default, getitem_813);  getitem_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_12: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_26, add_704, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_329)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1111: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_27, max_pool2d_with_indices_backward_12);  slice_27 = max_pool2d_with_indices_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_120: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_26, [0, 2, 3])
    sub_437: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_262, unsqueeze_1514);  convolution_262 = unsqueeze_1514 = None
    mul_1938: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_26, sub_437)
    sum_121: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1938, [0, 2, 3]);  mul_1938 = None
    mul_1939: "f32[432]" = torch.ops.aten.mul.Tensor(sum_120, 0.0002834467120181406)
    unsqueeze_1515: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1939, 0);  mul_1939 = None
    unsqueeze_1516: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
    unsqueeze_1517: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
    mul_1940: "f32[432]" = torch.ops.aten.mul.Tensor(sum_121, 0.0002834467120181406)
    mul_1941: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_424, squeeze_424)
    mul_1942: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1940, mul_1941);  mul_1940 = mul_1941 = None
    unsqueeze_1518: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1942, 0);  mul_1942 = None
    unsqueeze_1519: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
    unsqueeze_1520: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
    mul_1943: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_424, primals_552);  primals_552 = None
    unsqueeze_1521: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1943, 0);  mul_1943 = None
    unsqueeze_1522: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
    unsqueeze_1523: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
    mul_1944: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1520);  sub_437 = unsqueeze_1520 = None
    sub_439: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_26, mul_1944);  slice_26 = mul_1944 = None
    sub_440: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_439, unsqueeze_1517);  sub_439 = unsqueeze_1517 = None
    mul_1945: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1523);  sub_440 = unsqueeze_1523 = None
    mul_1946: "f32[432]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_424);  sum_121 = squeeze_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1945, convolution_261, primals_551, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1945 = convolution_261 = primals_551 = None
    getitem_816: "f32[8, 432, 21, 21]" = convolution_backward_110[0]
    getitem_817: "f32[432, 432, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(getitem_816, relu_139, primals_550, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_816 = primals_550 = None
    getitem_819: "f32[8, 432, 21, 21]" = convolution_backward_111[0]
    getitem_820: "f32[432, 1, 3, 3]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_60: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_139, 0);  relu_139 = None
    where_60: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_60, full_default, getitem_819);  le_60 = getitem_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_122: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_441: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_260, unsqueeze_1526);  convolution_260 = unsqueeze_1526 = None
    mul_1947: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_60, sub_441)
    sum_123: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1947, [0, 2, 3]);  mul_1947 = None
    mul_1948: "f32[432]" = torch.ops.aten.mul.Tensor(sum_122, 0.0002834467120181406)
    unsqueeze_1527: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1948, 0);  mul_1948 = None
    unsqueeze_1528: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
    unsqueeze_1529: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
    mul_1949: "f32[432]" = torch.ops.aten.mul.Tensor(sum_123, 0.0002834467120181406)
    mul_1950: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_421, squeeze_421)
    mul_1951: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1949, mul_1950);  mul_1949 = mul_1950 = None
    unsqueeze_1530: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1951, 0);  mul_1951 = None
    unsqueeze_1531: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
    unsqueeze_1532: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
    mul_1952: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_421, primals_548);  primals_548 = None
    unsqueeze_1533: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1952, 0);  mul_1952 = None
    unsqueeze_1534: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
    unsqueeze_1535: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
    mul_1953: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1532);  sub_441 = unsqueeze_1532 = None
    sub_443: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_60, mul_1953);  where_60 = mul_1953 = None
    sub_444: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_443, unsqueeze_1529);  sub_443 = unsqueeze_1529 = None
    mul_1954: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1535);  sub_444 = unsqueeze_1535 = None
    mul_1955: "f32[432]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_421);  sum_123 = squeeze_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_1954, convolution_259, primals_547, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1954 = convolution_259 = primals_547 = None
    getitem_822: "f32[8, 432, 21, 21]" = convolution_backward_112[0]
    getitem_823: "f32[432, 432, 1, 1]" = convolution_backward_112[1];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(getitem_822, relu_138, primals_546, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_822 = primals_546 = None
    getitem_825: "f32[8, 432, 21, 21]" = convolution_backward_113[0]
    getitem_826: "f32[432, 1, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_61: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_138, 0);  relu_138 = None
    where_61: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_61, full_default, getitem_825);  le_61 = getitem_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1112: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_25, where_61);  slice_25 = where_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_124: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1112, [0, 2, 3])
    sub_445: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_258, unsqueeze_1538);  convolution_258 = unsqueeze_1538 = None
    mul_1956: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1112, sub_445)
    sum_125: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1956, [0, 2, 3]);  mul_1956 = None
    mul_1957: "f32[432]" = torch.ops.aten.mul.Tensor(sum_124, 0.0002834467120181406)
    unsqueeze_1539: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1957, 0);  mul_1957 = None
    unsqueeze_1540: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 2);  unsqueeze_1539 = None
    unsqueeze_1541: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 3);  unsqueeze_1540 = None
    mul_1958: "f32[432]" = torch.ops.aten.mul.Tensor(sum_125, 0.0002834467120181406)
    mul_1959: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_418, squeeze_418)
    mul_1960: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1958, mul_1959);  mul_1958 = mul_1959 = None
    unsqueeze_1542: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1960, 0);  mul_1960 = None
    unsqueeze_1543: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 2);  unsqueeze_1542 = None
    unsqueeze_1544: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 3);  unsqueeze_1543 = None
    mul_1961: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_418, primals_544);  primals_544 = None
    unsqueeze_1545: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1961, 0);  mul_1961 = None
    unsqueeze_1546: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 2);  unsqueeze_1545 = None
    unsqueeze_1547: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 3);  unsqueeze_1546 = None
    mul_1962: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1544);  sub_445 = unsqueeze_1544 = None
    sub_447: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1112, mul_1962);  mul_1962 = None
    sub_448: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_447, unsqueeze_1541);  sub_447 = None
    mul_1963: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1547);  sub_448 = unsqueeze_1547 = None
    mul_1964: "f32[432]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_418);  sum_125 = squeeze_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_1963, convolution_257, primals_543, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1963 = convolution_257 = primals_543 = None
    getitem_828: "f32[8, 432, 21, 21]" = convolution_backward_114[0]
    getitem_829: "f32[432, 432, 1, 1]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(getitem_828, relu_137, primals_542, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_828 = primals_542 = None
    getitem_831: "f32[8, 432, 21, 21]" = convolution_backward_115[0]
    getitem_832: "f32[432, 1, 3, 3]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_62: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_137, 0);  relu_137 = None
    where_62: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_62, full_default, getitem_831);  le_62 = getitem_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_126: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_449: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_256, unsqueeze_1550);  convolution_256 = unsqueeze_1550 = None
    mul_1965: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_62, sub_449)
    sum_127: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1965, [0, 2, 3]);  mul_1965 = None
    mul_1966: "f32[432]" = torch.ops.aten.mul.Tensor(sum_126, 0.0002834467120181406)
    unsqueeze_1551: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1966, 0);  mul_1966 = None
    unsqueeze_1552: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 2);  unsqueeze_1551 = None
    unsqueeze_1553: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 3);  unsqueeze_1552 = None
    mul_1967: "f32[432]" = torch.ops.aten.mul.Tensor(sum_127, 0.0002834467120181406)
    mul_1968: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_415, squeeze_415)
    mul_1969: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1967, mul_1968);  mul_1967 = mul_1968 = None
    unsqueeze_1554: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1969, 0);  mul_1969 = None
    unsqueeze_1555: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 2);  unsqueeze_1554 = None
    unsqueeze_1556: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 3);  unsqueeze_1555 = None
    mul_1970: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_415, primals_540);  primals_540 = None
    unsqueeze_1557: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1970, 0);  mul_1970 = None
    unsqueeze_1558: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 2);  unsqueeze_1557 = None
    unsqueeze_1559: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 3);  unsqueeze_1558 = None
    mul_1971: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1556);  sub_449 = unsqueeze_1556 = None
    sub_451: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_62, mul_1971);  where_62 = mul_1971 = None
    sub_452: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_451, unsqueeze_1553);  sub_451 = unsqueeze_1553 = None
    mul_1972: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1559);  sub_452 = unsqueeze_1559 = None
    mul_1973: "f32[432]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_415);  sum_127 = squeeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_1972, convolution_255, primals_539, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1972 = convolution_255 = primals_539 = None
    getitem_834: "f32[8, 432, 21, 21]" = convolution_backward_116[0]
    getitem_835: "f32[432, 432, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(getitem_834, relu_132, primals_538, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_834 = primals_538 = None
    getitem_837: "f32[8, 432, 21, 21]" = convolution_backward_117[0]
    getitem_838: "f32[432, 1, 3, 3]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_63: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_132, 0)
    where_63: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_63, full_default, getitem_837);  getitem_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1113: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1111, where_63);  add_1111 = where_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_453: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_254, unsqueeze_1562);  convolution_254 = unsqueeze_1562 = None
    mul_1974: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1112, sub_453)
    sum_129: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1974, [0, 2, 3]);  mul_1974 = None
    mul_1976: "f32[432]" = torch.ops.aten.mul.Tensor(sum_129, 0.0002834467120181406)
    mul_1977: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_412, squeeze_412)
    mul_1978: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1976, mul_1977);  mul_1976 = mul_1977 = None
    unsqueeze_1566: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1978, 0);  mul_1978 = None
    unsqueeze_1567: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 2);  unsqueeze_1566 = None
    unsqueeze_1568: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 3);  unsqueeze_1567 = None
    mul_1979: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_412, primals_536);  primals_536 = None
    unsqueeze_1569: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1979, 0);  mul_1979 = None
    unsqueeze_1570: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 2);  unsqueeze_1569 = None
    unsqueeze_1571: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 3);  unsqueeze_1570 = None
    mul_1980: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1568);  sub_453 = unsqueeze_1568 = None
    sub_455: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1112, mul_1980);  add_1112 = mul_1980 = None
    sub_456: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1541);  sub_455 = unsqueeze_1541 = None
    mul_1981: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1571);  sub_456 = unsqueeze_1571 = None
    mul_1982: "f32[432]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_412);  sum_129 = squeeze_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1981, convolution_253, primals_535, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1981 = convolution_253 = primals_535 = None
    getitem_840: "f32[8, 432, 21, 21]" = convolution_backward_118[0]
    getitem_841: "f32[432, 432, 1, 1]" = convolution_backward_118[1];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(getitem_840, relu_135, primals_534, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_840 = primals_534 = None
    getitem_843: "f32[8, 432, 21, 21]" = convolution_backward_119[0]
    getitem_844: "f32[432, 1, 5, 5]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_64: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_135, 0);  relu_135 = None
    where_64: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_64, full_default, getitem_843);  le_64 = getitem_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_130: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_457: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_252, unsqueeze_1574);  convolution_252 = unsqueeze_1574 = None
    mul_1983: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_64, sub_457)
    sum_131: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1983, [0, 2, 3]);  mul_1983 = None
    mul_1984: "f32[432]" = torch.ops.aten.mul.Tensor(sum_130, 0.0002834467120181406)
    unsqueeze_1575: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1984, 0);  mul_1984 = None
    unsqueeze_1576: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 2);  unsqueeze_1575 = None
    unsqueeze_1577: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 3);  unsqueeze_1576 = None
    mul_1985: "f32[432]" = torch.ops.aten.mul.Tensor(sum_131, 0.0002834467120181406)
    mul_1986: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_409, squeeze_409)
    mul_1987: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1985, mul_1986);  mul_1985 = mul_1986 = None
    unsqueeze_1578: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1987, 0);  mul_1987 = None
    unsqueeze_1579: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 2);  unsqueeze_1578 = None
    unsqueeze_1580: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 3);  unsqueeze_1579 = None
    mul_1988: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_409, primals_532);  primals_532 = None
    unsqueeze_1581: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1988, 0);  mul_1988 = None
    unsqueeze_1582: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 2);  unsqueeze_1581 = None
    unsqueeze_1583: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 3);  unsqueeze_1582 = None
    mul_1989: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1580);  sub_457 = unsqueeze_1580 = None
    sub_459: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_64, mul_1989);  where_64 = mul_1989 = None
    sub_460: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1577);  sub_459 = unsqueeze_1577 = None
    mul_1990: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1583);  sub_460 = unsqueeze_1583 = None
    mul_1991: "f32[432]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_409);  sum_131 = squeeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_1990, convolution_251, primals_531, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1990 = convolution_251 = primals_531 = None
    getitem_846: "f32[8, 432, 21, 21]" = convolution_backward_120[0]
    getitem_847: "f32[432, 432, 1, 1]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(getitem_846, relu_132, primals_530, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_846 = primals_530 = None
    getitem_849: "f32[8, 432, 21, 21]" = convolution_backward_121[0]
    getitem_850: "f32[432, 1, 5, 5]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_65: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_63, full_default, getitem_849);  getitem_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1114: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1113, where_65);  add_1113 = where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_13: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_24, add_704, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_329);  add_704 = getitem_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1115: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1114, max_pool2d_with_indices_backward_13);  add_1114 = max_pool2d_with_indices_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_132: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_24, [0, 2, 3])
    sub_461: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_250, unsqueeze_1586);  convolution_250 = unsqueeze_1586 = None
    mul_1992: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_24, sub_461)
    sum_133: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_1992, [0, 2, 3]);  mul_1992 = None
    mul_1993: "f32[432]" = torch.ops.aten.mul.Tensor(sum_132, 0.0002834467120181406)
    unsqueeze_1587: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1993, 0);  mul_1993 = None
    unsqueeze_1588: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 2);  unsqueeze_1587 = None
    unsqueeze_1589: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 3);  unsqueeze_1588 = None
    mul_1994: "f32[432]" = torch.ops.aten.mul.Tensor(sum_133, 0.0002834467120181406)
    mul_1995: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_406, squeeze_406)
    mul_1996: "f32[432]" = torch.ops.aten.mul.Tensor(mul_1994, mul_1995);  mul_1994 = mul_1995 = None
    unsqueeze_1590: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1996, 0);  mul_1996 = None
    unsqueeze_1591: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 2);  unsqueeze_1590 = None
    unsqueeze_1592: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 3);  unsqueeze_1591 = None
    mul_1997: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_406, primals_528);  primals_528 = None
    unsqueeze_1593: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_1997, 0);  mul_1997 = None
    unsqueeze_1594: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 2);  unsqueeze_1593 = None
    unsqueeze_1595: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 3);  unsqueeze_1594 = None
    mul_1998: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1592);  sub_461 = unsqueeze_1592 = None
    sub_463: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_24, mul_1998);  slice_24 = mul_1998 = None
    sub_464: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1589);  sub_463 = unsqueeze_1589 = None
    mul_1999: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1595);  sub_464 = unsqueeze_1595 = None
    mul_2000: "f32[432]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_406);  sum_133 = squeeze_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(mul_1999, convolution_249, primals_527, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1999 = convolution_249 = primals_527 = None
    getitem_852: "f32[8, 432, 21, 21]" = convolution_backward_122[0]
    getitem_853: "f32[432, 432, 1, 1]" = convolution_backward_122[1];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(getitem_852, relu_133, primals_526, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_852 = primals_526 = None
    getitem_855: "f32[8, 432, 21, 21]" = convolution_backward_123[0]
    getitem_856: "f32[432, 1, 7, 7]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_66: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_133, 0);  relu_133 = None
    where_66: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_66, full_default, getitem_855);  le_66 = getitem_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_134: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_465: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_248, unsqueeze_1598);  convolution_248 = unsqueeze_1598 = None
    mul_2001: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_66, sub_465)
    sum_135: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2001, [0, 2, 3]);  mul_2001 = None
    mul_2002: "f32[432]" = torch.ops.aten.mul.Tensor(sum_134, 0.0002834467120181406)
    unsqueeze_1599: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2002, 0);  mul_2002 = None
    unsqueeze_1600: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 2);  unsqueeze_1599 = None
    unsqueeze_1601: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 3);  unsqueeze_1600 = None
    mul_2003: "f32[432]" = torch.ops.aten.mul.Tensor(sum_135, 0.0002834467120181406)
    mul_2004: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_403, squeeze_403)
    mul_2005: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2003, mul_2004);  mul_2003 = mul_2004 = None
    unsqueeze_1602: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2005, 0);  mul_2005 = None
    unsqueeze_1603: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 2);  unsqueeze_1602 = None
    unsqueeze_1604: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 3);  unsqueeze_1603 = None
    mul_2006: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_403, primals_524);  primals_524 = None
    unsqueeze_1605: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2006, 0);  mul_2006 = None
    unsqueeze_1606: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 2);  unsqueeze_1605 = None
    unsqueeze_1607: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 3);  unsqueeze_1606 = None
    mul_2007: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1604);  sub_465 = unsqueeze_1604 = None
    sub_467: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_66, mul_2007);  where_66 = mul_2007 = None
    sub_468: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1601);  sub_467 = unsqueeze_1601 = None
    mul_2008: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1607);  sub_468 = unsqueeze_1607 = None
    mul_2009: "f32[432]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_403);  sum_135 = squeeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(mul_2008, convolution_247, primals_523, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2008 = convolution_247 = primals_523 = None
    getitem_858: "f32[8, 432, 21, 21]" = convolution_backward_124[0]
    getitem_859: "f32[432, 432, 1, 1]" = convolution_backward_124[1];  convolution_backward_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(getitem_858, relu_132, primals_522, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_858 = relu_132 = primals_522 = None
    getitem_861: "f32[8, 432, 21, 21]" = convolution_backward_125[0]
    getitem_862: "f32[432, 1, 7, 7]" = convolution_backward_125[1];  convolution_backward_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_67: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_63, full_default, getitem_861);  le_63 = getitem_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1116: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1115, where_67);  add_1115 = where_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_14: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_23, add_699, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_323);  add_699 = getitem_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1117: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(where_59, max_pool2d_with_indices_backward_14);  where_59 = max_pool2d_with_indices_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_136: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_23, [0, 2, 3])
    sub_469: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_246, unsqueeze_1610);  convolution_246 = unsqueeze_1610 = None
    mul_2010: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_23, sub_469)
    sum_137: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2010, [0, 2, 3]);  mul_2010 = None
    mul_2011: "f32[432]" = torch.ops.aten.mul.Tensor(sum_136, 0.0002834467120181406)
    unsqueeze_1611: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2011, 0);  mul_2011 = None
    unsqueeze_1612: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 2);  unsqueeze_1611 = None
    unsqueeze_1613: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 3);  unsqueeze_1612 = None
    mul_2012: "f32[432]" = torch.ops.aten.mul.Tensor(sum_137, 0.0002834467120181406)
    mul_2013: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_400, squeeze_400)
    mul_2014: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2012, mul_2013);  mul_2012 = mul_2013 = None
    unsqueeze_1614: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2014, 0);  mul_2014 = None
    unsqueeze_1615: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 2);  unsqueeze_1614 = None
    unsqueeze_1616: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 3);  unsqueeze_1615 = None
    mul_2015: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_400, primals_520);  primals_520 = None
    unsqueeze_1617: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2015, 0);  mul_2015 = None
    unsqueeze_1618: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 2);  unsqueeze_1617 = None
    unsqueeze_1619: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 3);  unsqueeze_1618 = None
    mul_2016: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1616);  sub_469 = unsqueeze_1616 = None
    sub_471: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_23, mul_2016);  slice_23 = mul_2016 = None
    sub_472: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_471, unsqueeze_1613);  sub_471 = unsqueeze_1613 = None
    mul_2017: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1619);  sub_472 = unsqueeze_1619 = None
    mul_2018: "f32[432]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_400);  sum_137 = squeeze_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_2017, convolution_245, primals_519, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2017 = convolution_245 = primals_519 = None
    getitem_864: "f32[8, 432, 21, 21]" = convolution_backward_126[0]
    getitem_865: "f32[432, 432, 1, 1]" = convolution_backward_126[1];  convolution_backward_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(getitem_864, relu_131, primals_518, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_864 = primals_518 = None
    getitem_867: "f32[8, 432, 21, 21]" = convolution_backward_127[0]
    getitem_868: "f32[432, 1, 5, 5]" = convolution_backward_127[1];  convolution_backward_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_68: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_131, 0);  relu_131 = None
    where_68: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_68, full_default, getitem_867);  le_68 = getitem_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_138: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_473: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_244, unsqueeze_1622);  convolution_244 = unsqueeze_1622 = None
    mul_2019: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_68, sub_473)
    sum_139: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2019, [0, 2, 3]);  mul_2019 = None
    mul_2020: "f32[432]" = torch.ops.aten.mul.Tensor(sum_138, 0.0002834467120181406)
    unsqueeze_1623: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2020, 0);  mul_2020 = None
    unsqueeze_1624: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 2);  unsqueeze_1623 = None
    unsqueeze_1625: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 3);  unsqueeze_1624 = None
    mul_2021: "f32[432]" = torch.ops.aten.mul.Tensor(sum_139, 0.0002834467120181406)
    mul_2022: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_397, squeeze_397)
    mul_2023: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2021, mul_2022);  mul_2021 = mul_2022 = None
    unsqueeze_1626: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2023, 0);  mul_2023 = None
    unsqueeze_1627: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 2);  unsqueeze_1626 = None
    unsqueeze_1628: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 3);  unsqueeze_1627 = None
    mul_2024: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_397, primals_516);  primals_516 = None
    unsqueeze_1629: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2024, 0);  mul_2024 = None
    unsqueeze_1630: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 2);  unsqueeze_1629 = None
    unsqueeze_1631: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 3);  unsqueeze_1630 = None
    mul_2025: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1628);  sub_473 = unsqueeze_1628 = None
    sub_475: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_68, mul_2025);  where_68 = mul_2025 = None
    sub_476: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_475, unsqueeze_1625);  sub_475 = unsqueeze_1625 = None
    mul_2026: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1631);  sub_476 = unsqueeze_1631 = None
    mul_2027: "f32[432]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_397);  sum_139 = squeeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(mul_2026, convolution_243, primals_515, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2026 = convolution_243 = primals_515 = None
    getitem_870: "f32[8, 432, 21, 21]" = convolution_backward_128[0]
    getitem_871: "f32[432, 432, 1, 1]" = convolution_backward_128[1];  convolution_backward_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(getitem_870, relu_130, primals_514, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_870 = relu_130 = primals_514 = None
    getitem_873: "f32[8, 432, 21, 21]" = convolution_backward_129[0]
    getitem_874: "f32[432, 1, 5, 5]" = convolution_backward_129[1];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_69: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_59, full_default, getitem_873);  le_59 = getitem_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1118: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1117, where_69);  add_1117 = where_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_140: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1116, [0, 2, 3])
    sub_477: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_242, unsqueeze_1634);  convolution_242 = unsqueeze_1634 = None
    mul_2028: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1116, sub_477)
    sum_141: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2028, [0, 2, 3]);  mul_2028 = None
    mul_2029: "f32[432]" = torch.ops.aten.mul.Tensor(sum_140, 0.0002834467120181406)
    unsqueeze_1635: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2029, 0);  mul_2029 = None
    unsqueeze_1636: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 2);  unsqueeze_1635 = None
    unsqueeze_1637: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, 3);  unsqueeze_1636 = None
    mul_2030: "f32[432]" = torch.ops.aten.mul.Tensor(sum_141, 0.0002834467120181406)
    mul_2031: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_394, squeeze_394)
    mul_2032: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2030, mul_2031);  mul_2030 = mul_2031 = None
    unsqueeze_1638: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2032, 0);  mul_2032 = None
    unsqueeze_1639: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 2);  unsqueeze_1638 = None
    unsqueeze_1640: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 3);  unsqueeze_1639 = None
    mul_2033: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_394, primals_512);  primals_512 = None
    unsqueeze_1641: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2033, 0);  mul_2033 = None
    unsqueeze_1642: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 2);  unsqueeze_1641 = None
    unsqueeze_1643: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 3);  unsqueeze_1642 = None
    mul_2034: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1640);  sub_477 = unsqueeze_1640 = None
    sub_479: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1116, mul_2034);  add_1116 = mul_2034 = None
    sub_480: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_479, unsqueeze_1637);  sub_479 = unsqueeze_1637 = None
    mul_2035: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1643);  sub_480 = unsqueeze_1643 = None
    mul_2036: "f32[432]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_394);  sum_141 = squeeze_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(mul_2035, relu_129, primals_511, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2035 = relu_129 = primals_511 = None
    getitem_876: "f32[8, 2160, 21, 21]" = convolution_backward_130[0]
    getitem_877: "f32[432, 2160, 1, 1]" = convolution_backward_130[1];  convolution_backward_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_70: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_57, full_default, getitem_876);  le_57 = getitem_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1119: "f32[8, 2160, 21, 21]" = torch.ops.aten.add.Tensor(where_57, where_70);  where_57 = where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_142: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1118, [0, 2, 3])
    sub_481: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_241, unsqueeze_1646);  convolution_241 = unsqueeze_1646 = None
    mul_2037: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1118, sub_481)
    sum_143: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2037, [0, 2, 3]);  mul_2037 = None
    mul_2038: "f32[432]" = torch.ops.aten.mul.Tensor(sum_142, 0.0002834467120181406)
    unsqueeze_1647: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2038, 0);  mul_2038 = None
    unsqueeze_1648: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 2);  unsqueeze_1647 = None
    unsqueeze_1649: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 3);  unsqueeze_1648 = None
    mul_2039: "f32[432]" = torch.ops.aten.mul.Tensor(sum_143, 0.0002834467120181406)
    mul_2040: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_391, squeeze_391)
    mul_2041: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2039, mul_2040);  mul_2039 = mul_2040 = None
    unsqueeze_1650: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2041, 0);  mul_2041 = None
    unsqueeze_1651: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 2);  unsqueeze_1650 = None
    unsqueeze_1652: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 3);  unsqueeze_1651 = None
    mul_2042: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_391, primals_509);  primals_509 = None
    unsqueeze_1653: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2042, 0);  mul_2042 = None
    unsqueeze_1654: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 2);  unsqueeze_1653 = None
    unsqueeze_1655: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 3);  unsqueeze_1654 = None
    mul_2043: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1652);  sub_481 = unsqueeze_1652 = None
    sub_483: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1118, mul_2043);  add_1118 = mul_2043 = None
    sub_484: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_483, unsqueeze_1649);  sub_483 = unsqueeze_1649 = None
    mul_2044: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1655);  sub_484 = unsqueeze_1655 = None
    mul_2045: "f32[432]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_391);  sum_143 = squeeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(mul_2044, relu_115, primals_508, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2044 = primals_508 = None
    getitem_879: "f32[8, 2160, 21, 21]" = convolution_backward_131[0]
    getitem_880: "f32[432, 2160, 1, 1]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_71: "b8[8, 2160, 21, 21]" = torch.ops.aten.le.Scalar(relu_115, 0)
    where_71: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_71, full_default, getitem_879);  getitem_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_28: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1119, 1, 0, 432)
    slice_29: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1119, 1, 432, 864)
    slice_30: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1119, 1, 864, 1296)
    slice_31: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1119, 1, 1296, 1728)
    slice_32: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1119, 1, 1728, 2160);  add_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_144: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_32, [0, 2, 3])
    sub_485: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_240, unsqueeze_1658);  convolution_240 = unsqueeze_1658 = None
    mul_2046: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_32, sub_485)
    sum_145: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2046, [0, 2, 3]);  mul_2046 = None
    mul_2047: "f32[432]" = torch.ops.aten.mul.Tensor(sum_144, 0.0002834467120181406)
    unsqueeze_1659: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2047, 0);  mul_2047 = None
    unsqueeze_1660: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 2);  unsqueeze_1659 = None
    unsqueeze_1661: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 3);  unsqueeze_1660 = None
    mul_2048: "f32[432]" = torch.ops.aten.mul.Tensor(sum_145, 0.0002834467120181406)
    mul_2049: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_388, squeeze_388)
    mul_2050: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2048, mul_2049);  mul_2048 = mul_2049 = None
    unsqueeze_1662: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2050, 0);  mul_2050 = None
    unsqueeze_1663: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 2);  unsqueeze_1662 = None
    unsqueeze_1664: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 3);  unsqueeze_1663 = None
    mul_2051: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_388, primals_506);  primals_506 = None
    unsqueeze_1665: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2051, 0);  mul_2051 = None
    unsqueeze_1666: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 2);  unsqueeze_1665 = None
    unsqueeze_1667: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 3);  unsqueeze_1666 = None
    mul_2052: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1664);  sub_485 = unsqueeze_1664 = None
    sub_487: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_32, mul_2052);  mul_2052 = None
    sub_488: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_487, unsqueeze_1661);  sub_487 = unsqueeze_1661 = None
    mul_2053: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1667);  sub_488 = unsqueeze_1667 = None
    mul_2054: "f32[432]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_388);  sum_145 = squeeze_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_2053, convolution_239, primals_505, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2053 = convolution_239 = primals_505 = None
    getitem_882: "f32[8, 432, 21, 21]" = convolution_backward_132[0]
    getitem_883: "f32[432, 432, 1, 1]" = convolution_backward_132[1];  convolution_backward_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(getitem_882, relu_127, primals_504, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_882 = primals_504 = None
    getitem_885: "f32[8, 432, 21, 21]" = convolution_backward_133[0]
    getitem_886: "f32[432, 1, 3, 3]" = convolution_backward_133[1];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_72: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_127, 0);  relu_127 = None
    where_72: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_72, full_default, getitem_885);  le_72 = getitem_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_146: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_489: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_238, unsqueeze_1670);  convolution_238 = unsqueeze_1670 = None
    mul_2055: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_72, sub_489)
    sum_147: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2055, [0, 2, 3]);  mul_2055 = None
    mul_2056: "f32[432]" = torch.ops.aten.mul.Tensor(sum_146, 0.0002834467120181406)
    unsqueeze_1671: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2056, 0);  mul_2056 = None
    unsqueeze_1672: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 2);  unsqueeze_1671 = None
    unsqueeze_1673: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 3);  unsqueeze_1672 = None
    mul_2057: "f32[432]" = torch.ops.aten.mul.Tensor(sum_147, 0.0002834467120181406)
    mul_2058: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_385, squeeze_385)
    mul_2059: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2057, mul_2058);  mul_2057 = mul_2058 = None
    unsqueeze_1674: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2059, 0);  mul_2059 = None
    unsqueeze_1675: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 2);  unsqueeze_1674 = None
    unsqueeze_1676: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 3);  unsqueeze_1675 = None
    mul_2060: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_385, primals_502);  primals_502 = None
    unsqueeze_1677: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2060, 0);  mul_2060 = None
    unsqueeze_1678: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 2);  unsqueeze_1677 = None
    unsqueeze_1679: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 3);  unsqueeze_1678 = None
    mul_2061: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1676);  sub_489 = unsqueeze_1676 = None
    sub_491: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_72, mul_2061);  where_72 = mul_2061 = None
    sub_492: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_491, unsqueeze_1673);  sub_491 = unsqueeze_1673 = None
    mul_2062: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1679);  sub_492 = unsqueeze_1679 = None
    mul_2063: "f32[432]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_385);  sum_147 = squeeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_2062, convolution_237, primals_501, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2062 = convolution_237 = primals_501 = None
    getitem_888: "f32[8, 432, 21, 21]" = convolution_backward_134[0]
    getitem_889: "f32[432, 432, 1, 1]" = convolution_backward_134[1];  convolution_backward_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(getitem_888, relu_116, primals_500, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_888 = primals_500 = None
    getitem_891: "f32[8, 432, 21, 21]" = convolution_backward_135[0]
    getitem_892: "f32[432, 1, 3, 3]" = convolution_backward_135[1];  convolution_backward_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_73: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_116, 0)
    where_73: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_73, full_default, getitem_891);  getitem_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_15: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_31, add_629, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_295)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1120: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_32, max_pool2d_with_indices_backward_15);  slice_32 = max_pool2d_with_indices_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_148: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_31, [0, 2, 3])
    sub_493: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_236, unsqueeze_1682);  convolution_236 = unsqueeze_1682 = None
    mul_2064: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_31, sub_493)
    sum_149: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2064, [0, 2, 3]);  mul_2064 = None
    mul_2065: "f32[432]" = torch.ops.aten.mul.Tensor(sum_148, 0.0002834467120181406)
    unsqueeze_1683: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2065, 0);  mul_2065 = None
    unsqueeze_1684: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1683, 2);  unsqueeze_1683 = None
    unsqueeze_1685: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, 3);  unsqueeze_1684 = None
    mul_2066: "f32[432]" = torch.ops.aten.mul.Tensor(sum_149, 0.0002834467120181406)
    mul_2067: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_382, squeeze_382)
    mul_2068: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2066, mul_2067);  mul_2066 = mul_2067 = None
    unsqueeze_1686: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2068, 0);  mul_2068 = None
    unsqueeze_1687: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 2);  unsqueeze_1686 = None
    unsqueeze_1688: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1687, 3);  unsqueeze_1687 = None
    mul_2069: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_382, primals_498);  primals_498 = None
    unsqueeze_1689: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2069, 0);  mul_2069 = None
    unsqueeze_1690: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1689, 2);  unsqueeze_1689 = None
    unsqueeze_1691: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, 3);  unsqueeze_1690 = None
    mul_2070: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1688);  sub_493 = unsqueeze_1688 = None
    sub_495: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_31, mul_2070);  slice_31 = mul_2070 = None
    sub_496: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_495, unsqueeze_1685);  sub_495 = unsqueeze_1685 = None
    mul_2071: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1691);  sub_496 = unsqueeze_1691 = None
    mul_2072: "f32[432]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_382);  sum_149 = squeeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(mul_2071, convolution_235, primals_497, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2071 = convolution_235 = primals_497 = None
    getitem_894: "f32[8, 432, 21, 21]" = convolution_backward_136[0]
    getitem_895: "f32[432, 432, 1, 1]" = convolution_backward_136[1];  convolution_backward_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(getitem_894, relu_125, primals_496, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_894 = primals_496 = None
    getitem_897: "f32[8, 432, 21, 21]" = convolution_backward_137[0]
    getitem_898: "f32[432, 1, 3, 3]" = convolution_backward_137[1];  convolution_backward_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_74: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_125, 0);  relu_125 = None
    where_74: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_74, full_default, getitem_897);  le_74 = getitem_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_150: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_497: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_234, unsqueeze_1694);  convolution_234 = unsqueeze_1694 = None
    mul_2073: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_74, sub_497)
    sum_151: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2073, [0, 2, 3]);  mul_2073 = None
    mul_2074: "f32[432]" = torch.ops.aten.mul.Tensor(sum_150, 0.0002834467120181406)
    unsqueeze_1695: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2074, 0);  mul_2074 = None
    unsqueeze_1696: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 2);  unsqueeze_1695 = None
    unsqueeze_1697: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, 3);  unsqueeze_1696 = None
    mul_2075: "f32[432]" = torch.ops.aten.mul.Tensor(sum_151, 0.0002834467120181406)
    mul_2076: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_379, squeeze_379)
    mul_2077: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2075, mul_2076);  mul_2075 = mul_2076 = None
    unsqueeze_1698: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2077, 0);  mul_2077 = None
    unsqueeze_1699: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 2);  unsqueeze_1698 = None
    unsqueeze_1700: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1699, 3);  unsqueeze_1699 = None
    mul_2078: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_379, primals_494);  primals_494 = None
    unsqueeze_1701: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2078, 0);  mul_2078 = None
    unsqueeze_1702: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 2);  unsqueeze_1701 = None
    unsqueeze_1703: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, 3);  unsqueeze_1702 = None
    mul_2079: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1700);  sub_497 = unsqueeze_1700 = None
    sub_499: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_74, mul_2079);  where_74 = mul_2079 = None
    sub_500: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_499, unsqueeze_1697);  sub_499 = unsqueeze_1697 = None
    mul_2080: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1703);  sub_500 = unsqueeze_1703 = None
    mul_2081: "f32[432]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_379);  sum_151 = squeeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(mul_2080, convolution_233, primals_493, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2080 = convolution_233 = primals_493 = None
    getitem_900: "f32[8, 432, 21, 21]" = convolution_backward_138[0]
    getitem_901: "f32[432, 432, 1, 1]" = convolution_backward_138[1];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(getitem_900, relu_124, primals_492, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_900 = primals_492 = None
    getitem_903: "f32[8, 432, 21, 21]" = convolution_backward_139[0]
    getitem_904: "f32[432, 1, 3, 3]" = convolution_backward_139[1];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_75: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_124, 0);  relu_124 = None
    where_75: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_75, full_default, getitem_903);  le_75 = getitem_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1121: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_30, where_75);  slice_30 = where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_152: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1121, [0, 2, 3])
    sub_501: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_232, unsqueeze_1706);  convolution_232 = unsqueeze_1706 = None
    mul_2082: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1121, sub_501)
    sum_153: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2082, [0, 2, 3]);  mul_2082 = None
    mul_2083: "f32[432]" = torch.ops.aten.mul.Tensor(sum_152, 0.0002834467120181406)
    unsqueeze_1707: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2083, 0);  mul_2083 = None
    unsqueeze_1708: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 2);  unsqueeze_1707 = None
    unsqueeze_1709: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, 3);  unsqueeze_1708 = None
    mul_2084: "f32[432]" = torch.ops.aten.mul.Tensor(sum_153, 0.0002834467120181406)
    mul_2085: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_376, squeeze_376)
    mul_2086: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2084, mul_2085);  mul_2084 = mul_2085 = None
    unsqueeze_1710: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2086, 0);  mul_2086 = None
    unsqueeze_1711: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 2);  unsqueeze_1710 = None
    unsqueeze_1712: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1711, 3);  unsqueeze_1711 = None
    mul_2087: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_376, primals_490);  primals_490 = None
    unsqueeze_1713: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2087, 0);  mul_2087 = None
    unsqueeze_1714: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 2);  unsqueeze_1713 = None
    unsqueeze_1715: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, 3);  unsqueeze_1714 = None
    mul_2088: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1712);  sub_501 = unsqueeze_1712 = None
    sub_503: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1121, mul_2088);  mul_2088 = None
    sub_504: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_503, unsqueeze_1709);  sub_503 = None
    mul_2089: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1715);  sub_504 = unsqueeze_1715 = None
    mul_2090: "f32[432]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_376);  sum_153 = squeeze_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(mul_2089, convolution_231, primals_489, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2089 = convolution_231 = primals_489 = None
    getitem_906: "f32[8, 432, 21, 21]" = convolution_backward_140[0]
    getitem_907: "f32[432, 432, 1, 1]" = convolution_backward_140[1];  convolution_backward_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(getitem_906, relu_123, primals_488, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_906 = primals_488 = None
    getitem_909: "f32[8, 432, 21, 21]" = convolution_backward_141[0]
    getitem_910: "f32[432, 1, 3, 3]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_76: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_123, 0);  relu_123 = None
    where_76: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_76, full_default, getitem_909);  le_76 = getitem_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_154: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_505: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_230, unsqueeze_1718);  convolution_230 = unsqueeze_1718 = None
    mul_2091: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_76, sub_505)
    sum_155: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2091, [0, 2, 3]);  mul_2091 = None
    mul_2092: "f32[432]" = torch.ops.aten.mul.Tensor(sum_154, 0.0002834467120181406)
    unsqueeze_1719: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2092, 0);  mul_2092 = None
    unsqueeze_1720: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 2);  unsqueeze_1719 = None
    unsqueeze_1721: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, 3);  unsqueeze_1720 = None
    mul_2093: "f32[432]" = torch.ops.aten.mul.Tensor(sum_155, 0.0002834467120181406)
    mul_2094: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_373, squeeze_373)
    mul_2095: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2093, mul_2094);  mul_2093 = mul_2094 = None
    unsqueeze_1722: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2095, 0);  mul_2095 = None
    unsqueeze_1723: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 2);  unsqueeze_1722 = None
    unsqueeze_1724: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1723, 3);  unsqueeze_1723 = None
    mul_2096: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_373, primals_486);  primals_486 = None
    unsqueeze_1725: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2096, 0);  mul_2096 = None
    unsqueeze_1726: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 2);  unsqueeze_1725 = None
    unsqueeze_1727: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, 3);  unsqueeze_1726 = None
    mul_2097: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1724);  sub_505 = unsqueeze_1724 = None
    sub_507: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_76, mul_2097);  where_76 = mul_2097 = None
    sub_508: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_507, unsqueeze_1721);  sub_507 = unsqueeze_1721 = None
    mul_2098: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1727);  sub_508 = unsqueeze_1727 = None
    mul_2099: "f32[432]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_373);  sum_155 = squeeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_2098, convolution_229, primals_485, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2098 = convolution_229 = primals_485 = None
    getitem_912: "f32[8, 432, 21, 21]" = convolution_backward_142[0]
    getitem_913: "f32[432, 432, 1, 1]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(getitem_912, relu_118, primals_484, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_912 = primals_484 = None
    getitem_915: "f32[8, 432, 21, 21]" = convolution_backward_143[0]
    getitem_916: "f32[432, 1, 3, 3]" = convolution_backward_143[1];  convolution_backward_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_77: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_118, 0)
    where_77: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_77, full_default, getitem_915);  getitem_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1122: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1120, where_77);  add_1120 = where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_509: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_228, unsqueeze_1730);  convolution_228 = unsqueeze_1730 = None
    mul_2100: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1121, sub_509)
    sum_157: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2100, [0, 2, 3]);  mul_2100 = None
    mul_2102: "f32[432]" = torch.ops.aten.mul.Tensor(sum_157, 0.0002834467120181406)
    mul_2103: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_370, squeeze_370)
    mul_2104: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2102, mul_2103);  mul_2102 = mul_2103 = None
    unsqueeze_1734: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2104, 0);  mul_2104 = None
    unsqueeze_1735: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 2);  unsqueeze_1734 = None
    unsqueeze_1736: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1735, 3);  unsqueeze_1735 = None
    mul_2105: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_370, primals_482);  primals_482 = None
    unsqueeze_1737: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2105, 0);  mul_2105 = None
    unsqueeze_1738: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 2);  unsqueeze_1737 = None
    unsqueeze_1739: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, 3);  unsqueeze_1738 = None
    mul_2106: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1736);  sub_509 = unsqueeze_1736 = None
    sub_511: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1121, mul_2106);  add_1121 = mul_2106 = None
    sub_512: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_511, unsqueeze_1709);  sub_511 = unsqueeze_1709 = None
    mul_2107: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1739);  sub_512 = unsqueeze_1739 = None
    mul_2108: "f32[432]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_370);  sum_157 = squeeze_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(mul_2107, convolution_227, primals_481, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2107 = convolution_227 = primals_481 = None
    getitem_918: "f32[8, 432, 21, 21]" = convolution_backward_144[0]
    getitem_919: "f32[432, 432, 1, 1]" = convolution_backward_144[1];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(getitem_918, relu_121, primals_480, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_918 = primals_480 = None
    getitem_921: "f32[8, 432, 21, 21]" = convolution_backward_145[0]
    getitem_922: "f32[432, 1, 5, 5]" = convolution_backward_145[1];  convolution_backward_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_78: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_121, 0);  relu_121 = None
    where_78: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_78, full_default, getitem_921);  le_78 = getitem_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_158: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_513: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_226, unsqueeze_1742);  convolution_226 = unsqueeze_1742 = None
    mul_2109: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_78, sub_513)
    sum_159: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2109, [0, 2, 3]);  mul_2109 = None
    mul_2110: "f32[432]" = torch.ops.aten.mul.Tensor(sum_158, 0.0002834467120181406)
    unsqueeze_1743: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2110, 0);  mul_2110 = None
    unsqueeze_1744: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1743, 2);  unsqueeze_1743 = None
    unsqueeze_1745: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, 3);  unsqueeze_1744 = None
    mul_2111: "f32[432]" = torch.ops.aten.mul.Tensor(sum_159, 0.0002834467120181406)
    mul_2112: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_367, squeeze_367)
    mul_2113: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2111, mul_2112);  mul_2111 = mul_2112 = None
    unsqueeze_1746: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2113, 0);  mul_2113 = None
    unsqueeze_1747: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 2);  unsqueeze_1746 = None
    unsqueeze_1748: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1747, 3);  unsqueeze_1747 = None
    mul_2114: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_367, primals_478);  primals_478 = None
    unsqueeze_1749: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2114, 0);  mul_2114 = None
    unsqueeze_1750: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1749, 2);  unsqueeze_1749 = None
    unsqueeze_1751: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, 3);  unsqueeze_1750 = None
    mul_2115: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1748);  sub_513 = unsqueeze_1748 = None
    sub_515: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_78, mul_2115);  where_78 = mul_2115 = None
    sub_516: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_515, unsqueeze_1745);  sub_515 = unsqueeze_1745 = None
    mul_2116: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1751);  sub_516 = unsqueeze_1751 = None
    mul_2117: "f32[432]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_367);  sum_159 = squeeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(mul_2116, convolution_225, primals_477, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2116 = convolution_225 = primals_477 = None
    getitem_924: "f32[8, 432, 21, 21]" = convolution_backward_146[0]
    getitem_925: "f32[432, 432, 1, 1]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(getitem_924, relu_118, primals_476, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_924 = primals_476 = None
    getitem_927: "f32[8, 432, 21, 21]" = convolution_backward_147[0]
    getitem_928: "f32[432, 1, 5, 5]" = convolution_backward_147[1];  convolution_backward_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_79: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_77, full_default, getitem_927);  getitem_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1123: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1122, where_79);  add_1122 = where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_16: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_29, add_629, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_295);  add_629 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1124: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1123, max_pool2d_with_indices_backward_16);  add_1123 = max_pool2d_with_indices_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_160: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_29, [0, 2, 3])
    sub_517: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_224, unsqueeze_1754);  convolution_224 = unsqueeze_1754 = None
    mul_2118: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_29, sub_517)
    sum_161: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2118, [0, 2, 3]);  mul_2118 = None
    mul_2119: "f32[432]" = torch.ops.aten.mul.Tensor(sum_160, 0.0002834467120181406)
    unsqueeze_1755: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2119, 0);  mul_2119 = None
    unsqueeze_1756: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1755, 2);  unsqueeze_1755 = None
    unsqueeze_1757: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, 3);  unsqueeze_1756 = None
    mul_2120: "f32[432]" = torch.ops.aten.mul.Tensor(sum_161, 0.0002834467120181406)
    mul_2121: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_364, squeeze_364)
    mul_2122: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2120, mul_2121);  mul_2120 = mul_2121 = None
    unsqueeze_1758: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2122, 0);  mul_2122 = None
    unsqueeze_1759: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 2);  unsqueeze_1758 = None
    unsqueeze_1760: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1759, 3);  unsqueeze_1759 = None
    mul_2123: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_364, primals_474);  primals_474 = None
    unsqueeze_1761: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2123, 0);  mul_2123 = None
    unsqueeze_1762: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1761, 2);  unsqueeze_1761 = None
    unsqueeze_1763: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, 3);  unsqueeze_1762 = None
    mul_2124: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1760);  sub_517 = unsqueeze_1760 = None
    sub_519: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_29, mul_2124);  slice_29 = mul_2124 = None
    sub_520: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_519, unsqueeze_1757);  sub_519 = unsqueeze_1757 = None
    mul_2125: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_1763);  sub_520 = unsqueeze_1763 = None
    mul_2126: "f32[432]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_364);  sum_161 = squeeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(mul_2125, convolution_223, primals_473, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2125 = convolution_223 = primals_473 = None
    getitem_930: "f32[8, 432, 21, 21]" = convolution_backward_148[0]
    getitem_931: "f32[432, 432, 1, 1]" = convolution_backward_148[1];  convolution_backward_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_149 = torch.ops.aten.convolution_backward.default(getitem_930, relu_119, primals_472, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_930 = primals_472 = None
    getitem_933: "f32[8, 432, 21, 21]" = convolution_backward_149[0]
    getitem_934: "f32[432, 1, 7, 7]" = convolution_backward_149[1];  convolution_backward_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_80: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_119, 0);  relu_119 = None
    where_80: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_80, full_default, getitem_933);  le_80 = getitem_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_162: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_521: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_222, unsqueeze_1766);  convolution_222 = unsqueeze_1766 = None
    mul_2127: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_80, sub_521)
    sum_163: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2127, [0, 2, 3]);  mul_2127 = None
    mul_2128: "f32[432]" = torch.ops.aten.mul.Tensor(sum_162, 0.0002834467120181406)
    unsqueeze_1767: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2128, 0);  mul_2128 = None
    unsqueeze_1768: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1767, 2);  unsqueeze_1767 = None
    unsqueeze_1769: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, 3);  unsqueeze_1768 = None
    mul_2129: "f32[432]" = torch.ops.aten.mul.Tensor(sum_163, 0.0002834467120181406)
    mul_2130: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
    mul_2131: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2129, mul_2130);  mul_2129 = mul_2130 = None
    unsqueeze_1770: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2131, 0);  mul_2131 = None
    unsqueeze_1771: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 2);  unsqueeze_1770 = None
    unsqueeze_1772: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1771, 3);  unsqueeze_1771 = None
    mul_2132: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_361, primals_470);  primals_470 = None
    unsqueeze_1773: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2132, 0);  mul_2132 = None
    unsqueeze_1774: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1773, 2);  unsqueeze_1773 = None
    unsqueeze_1775: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, 3);  unsqueeze_1774 = None
    mul_2133: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1772);  sub_521 = unsqueeze_1772 = None
    sub_523: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_80, mul_2133);  where_80 = mul_2133 = None
    sub_524: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1769);  sub_523 = unsqueeze_1769 = None
    mul_2134: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1775);  sub_524 = unsqueeze_1775 = None
    mul_2135: "f32[432]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_361);  sum_163 = squeeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_150 = torch.ops.aten.convolution_backward.default(mul_2134, convolution_221, primals_469, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2134 = convolution_221 = primals_469 = None
    getitem_936: "f32[8, 432, 21, 21]" = convolution_backward_150[0]
    getitem_937: "f32[432, 432, 1, 1]" = convolution_backward_150[1];  convolution_backward_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_151 = torch.ops.aten.convolution_backward.default(getitem_936, relu_118, primals_468, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_936 = relu_118 = primals_468 = None
    getitem_939: "f32[8, 432, 21, 21]" = convolution_backward_151[0]
    getitem_940: "f32[432, 1, 7, 7]" = convolution_backward_151[1];  convolution_backward_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_81: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_77, full_default, getitem_939);  le_77 = getitem_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1125: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1124, where_81);  add_1124 = where_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_17: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_28, add_624, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_289);  add_624 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1126: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(where_73, max_pool2d_with_indices_backward_17);  where_73 = max_pool2d_with_indices_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_164: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_28, [0, 2, 3])
    sub_525: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_220, unsqueeze_1778);  convolution_220 = unsqueeze_1778 = None
    mul_2136: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_28, sub_525)
    sum_165: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2136, [0, 2, 3]);  mul_2136 = None
    mul_2137: "f32[432]" = torch.ops.aten.mul.Tensor(sum_164, 0.0002834467120181406)
    unsqueeze_1779: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2137, 0);  mul_2137 = None
    unsqueeze_1780: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1779, 2);  unsqueeze_1779 = None
    unsqueeze_1781: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, 3);  unsqueeze_1780 = None
    mul_2138: "f32[432]" = torch.ops.aten.mul.Tensor(sum_165, 0.0002834467120181406)
    mul_2139: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
    mul_2140: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2138, mul_2139);  mul_2138 = mul_2139 = None
    unsqueeze_1782: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2140, 0);  mul_2140 = None
    unsqueeze_1783: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 2);  unsqueeze_1782 = None
    unsqueeze_1784: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1783, 3);  unsqueeze_1783 = None
    mul_2141: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_358, primals_466);  primals_466 = None
    unsqueeze_1785: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2141, 0);  mul_2141 = None
    unsqueeze_1786: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1785, 2);  unsqueeze_1785 = None
    unsqueeze_1787: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, 3);  unsqueeze_1786 = None
    mul_2142: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_1784);  sub_525 = unsqueeze_1784 = None
    sub_527: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_28, mul_2142);  slice_28 = mul_2142 = None
    sub_528: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_527, unsqueeze_1781);  sub_527 = unsqueeze_1781 = None
    mul_2143: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_528, unsqueeze_1787);  sub_528 = unsqueeze_1787 = None
    mul_2144: "f32[432]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_358);  sum_165 = squeeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_152 = torch.ops.aten.convolution_backward.default(mul_2143, convolution_219, primals_465, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2143 = convolution_219 = primals_465 = None
    getitem_942: "f32[8, 432, 21, 21]" = convolution_backward_152[0]
    getitem_943: "f32[432, 432, 1, 1]" = convolution_backward_152[1];  convolution_backward_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_153 = torch.ops.aten.convolution_backward.default(getitem_942, relu_117, primals_464, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_942 = primals_464 = None
    getitem_945: "f32[8, 432, 21, 21]" = convolution_backward_153[0]
    getitem_946: "f32[432, 1, 5, 5]" = convolution_backward_153[1];  convolution_backward_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_82: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_117, 0);  relu_117 = None
    where_82: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_82, full_default, getitem_945);  le_82 = getitem_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_166: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_529: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_218, unsqueeze_1790);  convolution_218 = unsqueeze_1790 = None
    mul_2145: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_82, sub_529)
    sum_167: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2145, [0, 2, 3]);  mul_2145 = None
    mul_2146: "f32[432]" = torch.ops.aten.mul.Tensor(sum_166, 0.0002834467120181406)
    unsqueeze_1791: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2146, 0);  mul_2146 = None
    unsqueeze_1792: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1791, 2);  unsqueeze_1791 = None
    unsqueeze_1793: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, 3);  unsqueeze_1792 = None
    mul_2147: "f32[432]" = torch.ops.aten.mul.Tensor(sum_167, 0.0002834467120181406)
    mul_2148: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
    mul_2149: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2147, mul_2148);  mul_2147 = mul_2148 = None
    unsqueeze_1794: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2149, 0);  mul_2149 = None
    unsqueeze_1795: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 2);  unsqueeze_1794 = None
    unsqueeze_1796: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1795, 3);  unsqueeze_1795 = None
    mul_2150: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_355, primals_462);  primals_462 = None
    unsqueeze_1797: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2150, 0);  mul_2150 = None
    unsqueeze_1798: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1797, 2);  unsqueeze_1797 = None
    unsqueeze_1799: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, 3);  unsqueeze_1798 = None
    mul_2151: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_1796);  sub_529 = unsqueeze_1796 = None
    sub_531: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_82, mul_2151);  where_82 = mul_2151 = None
    sub_532: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_531, unsqueeze_1793);  sub_531 = unsqueeze_1793 = None
    mul_2152: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_532, unsqueeze_1799);  sub_532 = unsqueeze_1799 = None
    mul_2153: "f32[432]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_355);  sum_167 = squeeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_154 = torch.ops.aten.convolution_backward.default(mul_2152, convolution_217, primals_461, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2152 = convolution_217 = primals_461 = None
    getitem_948: "f32[8, 432, 21, 21]" = convolution_backward_154[0]
    getitem_949: "f32[432, 432, 1, 1]" = convolution_backward_154[1];  convolution_backward_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_155 = torch.ops.aten.convolution_backward.default(getitem_948, relu_116, primals_460, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_948 = relu_116 = primals_460 = None
    getitem_951: "f32[8, 432, 21, 21]" = convolution_backward_155[0]
    getitem_952: "f32[432, 1, 5, 5]" = convolution_backward_155[1];  convolution_backward_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_83: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_73, full_default, getitem_951);  le_73 = getitem_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1127: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1126, where_83);  add_1126 = where_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_168: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1125, [0, 2, 3])
    sub_533: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_216, unsqueeze_1802);  convolution_216 = unsqueeze_1802 = None
    mul_2154: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1125, sub_533)
    sum_169: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2154, [0, 2, 3]);  mul_2154 = None
    mul_2155: "f32[432]" = torch.ops.aten.mul.Tensor(sum_168, 0.0002834467120181406)
    unsqueeze_1803: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2155, 0);  mul_2155 = None
    unsqueeze_1804: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1803, 2);  unsqueeze_1803 = None
    unsqueeze_1805: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, 3);  unsqueeze_1804 = None
    mul_2156: "f32[432]" = torch.ops.aten.mul.Tensor(sum_169, 0.0002834467120181406)
    mul_2157: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
    mul_2158: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2156, mul_2157);  mul_2156 = mul_2157 = None
    unsqueeze_1806: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2158, 0);  mul_2158 = None
    unsqueeze_1807: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 2);  unsqueeze_1806 = None
    unsqueeze_1808: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1807, 3);  unsqueeze_1807 = None
    mul_2159: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_352, primals_458);  primals_458 = None
    unsqueeze_1809: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2159, 0);  mul_2159 = None
    unsqueeze_1810: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1809, 2);  unsqueeze_1809 = None
    unsqueeze_1811: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, 3);  unsqueeze_1810 = None
    mul_2160: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_1808);  sub_533 = unsqueeze_1808 = None
    sub_535: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1125, mul_2160);  add_1125 = mul_2160 = None
    sub_536: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_535, unsqueeze_1805);  sub_535 = unsqueeze_1805 = None
    mul_2161: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_536, unsqueeze_1811);  sub_536 = unsqueeze_1811 = None
    mul_2162: "f32[432]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_352);  sum_169 = squeeze_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_156 = torch.ops.aten.convolution_backward.default(mul_2161, relu_115, primals_457, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2161 = relu_115 = primals_457 = None
    getitem_954: "f32[8, 2160, 21, 21]" = convolution_backward_156[0]
    getitem_955: "f32[432, 2160, 1, 1]" = convolution_backward_156[1];  convolution_backward_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_84: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_71, full_default, getitem_954);  le_71 = getitem_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1128: "f32[8, 2160, 21, 21]" = torch.ops.aten.add.Tensor(where_71, where_84);  where_71 = where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_170: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1127, [0, 2, 3])
    sub_537: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_215, unsqueeze_1814);  convolution_215 = unsqueeze_1814 = None
    mul_2163: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1127, sub_537)
    sum_171: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2163, [0, 2, 3]);  mul_2163 = None
    mul_2164: "f32[432]" = torch.ops.aten.mul.Tensor(sum_170, 0.0002834467120181406)
    unsqueeze_1815: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2164, 0);  mul_2164 = None
    unsqueeze_1816: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1815, 2);  unsqueeze_1815 = None
    unsqueeze_1817: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, 3);  unsqueeze_1816 = None
    mul_2165: "f32[432]" = torch.ops.aten.mul.Tensor(sum_171, 0.0002834467120181406)
    mul_2166: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
    mul_2167: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2165, mul_2166);  mul_2165 = mul_2166 = None
    unsqueeze_1818: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2167, 0);  mul_2167 = None
    unsqueeze_1819: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 2);  unsqueeze_1818 = None
    unsqueeze_1820: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1819, 3);  unsqueeze_1819 = None
    mul_2168: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_349, primals_455);  primals_455 = None
    unsqueeze_1821: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2168, 0);  mul_2168 = None
    unsqueeze_1822: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1821, 2);  unsqueeze_1821 = None
    unsqueeze_1823: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, 3);  unsqueeze_1822 = None
    mul_2169: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_537, unsqueeze_1820);  sub_537 = unsqueeze_1820 = None
    sub_539: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1127, mul_2169);  add_1127 = mul_2169 = None
    sub_540: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_539, unsqueeze_1817);  sub_539 = unsqueeze_1817 = None
    mul_2170: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_540, unsqueeze_1823);  sub_540 = unsqueeze_1823 = None
    mul_2171: "f32[432]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_349);  sum_171 = squeeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_157 = torch.ops.aten.convolution_backward.default(mul_2170, relu_101, primals_454, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2170 = primals_454 = None
    getitem_957: "f32[8, 2160, 21, 21]" = convolution_backward_157[0]
    getitem_958: "f32[432, 2160, 1, 1]" = convolution_backward_157[1];  convolution_backward_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_85: "b8[8, 2160, 21, 21]" = torch.ops.aten.le.Scalar(relu_101, 0)
    where_85: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_85, full_default, getitem_957);  getitem_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_33: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1128, 1, 0, 432)
    slice_34: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1128, 1, 432, 864)
    slice_35: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1128, 1, 864, 1296)
    slice_36: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1128, 1, 1296, 1728)
    slice_37: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1128, 1, 1728, 2160);  add_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_172: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_37, [0, 2, 3])
    sub_541: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_214, unsqueeze_1826);  convolution_214 = unsqueeze_1826 = None
    mul_2172: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_37, sub_541)
    sum_173: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2172, [0, 2, 3]);  mul_2172 = None
    mul_2173: "f32[432]" = torch.ops.aten.mul.Tensor(sum_172, 0.0002834467120181406)
    unsqueeze_1827: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2173, 0);  mul_2173 = None
    unsqueeze_1828: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1827, 2);  unsqueeze_1827 = None
    unsqueeze_1829: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, 3);  unsqueeze_1828 = None
    mul_2174: "f32[432]" = torch.ops.aten.mul.Tensor(sum_173, 0.0002834467120181406)
    mul_2175: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
    mul_2176: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2174, mul_2175);  mul_2174 = mul_2175 = None
    unsqueeze_1830: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2176, 0);  mul_2176 = None
    unsqueeze_1831: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 2);  unsqueeze_1830 = None
    unsqueeze_1832: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1831, 3);  unsqueeze_1831 = None
    mul_2177: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_346, primals_452);  primals_452 = None
    unsqueeze_1833: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2177, 0);  mul_2177 = None
    unsqueeze_1834: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1833, 2);  unsqueeze_1833 = None
    unsqueeze_1835: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, 3);  unsqueeze_1834 = None
    mul_2178: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_1832);  sub_541 = unsqueeze_1832 = None
    sub_543: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_37, mul_2178);  mul_2178 = None
    sub_544: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_543, unsqueeze_1829);  sub_543 = unsqueeze_1829 = None
    mul_2179: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_544, unsqueeze_1835);  sub_544 = unsqueeze_1835 = None
    mul_2180: "f32[432]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_346);  sum_173 = squeeze_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_158 = torch.ops.aten.convolution_backward.default(mul_2179, convolution_213, primals_451, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2179 = convolution_213 = primals_451 = None
    getitem_960: "f32[8, 432, 21, 21]" = convolution_backward_158[0]
    getitem_961: "f32[432, 432, 1, 1]" = convolution_backward_158[1];  convolution_backward_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_159 = torch.ops.aten.convolution_backward.default(getitem_960, relu_113, primals_450, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_960 = primals_450 = None
    getitem_963: "f32[8, 432, 21, 21]" = convolution_backward_159[0]
    getitem_964: "f32[432, 1, 3, 3]" = convolution_backward_159[1];  convolution_backward_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_86: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_113, 0);  relu_113 = None
    where_86: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_86, full_default, getitem_963);  le_86 = getitem_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_174: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_545: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_212, unsqueeze_1838);  convolution_212 = unsqueeze_1838 = None
    mul_2181: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_86, sub_545)
    sum_175: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2181, [0, 2, 3]);  mul_2181 = None
    mul_2182: "f32[432]" = torch.ops.aten.mul.Tensor(sum_174, 0.0002834467120181406)
    unsqueeze_1839: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2182, 0);  mul_2182 = None
    unsqueeze_1840: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1839, 2);  unsqueeze_1839 = None
    unsqueeze_1841: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, 3);  unsqueeze_1840 = None
    mul_2183: "f32[432]" = torch.ops.aten.mul.Tensor(sum_175, 0.0002834467120181406)
    mul_2184: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
    mul_2185: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2183, mul_2184);  mul_2183 = mul_2184 = None
    unsqueeze_1842: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2185, 0);  mul_2185 = None
    unsqueeze_1843: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, 2);  unsqueeze_1842 = None
    unsqueeze_1844: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1843, 3);  unsqueeze_1843 = None
    mul_2186: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_343, primals_448);  primals_448 = None
    unsqueeze_1845: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2186, 0);  mul_2186 = None
    unsqueeze_1846: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1845, 2);  unsqueeze_1845 = None
    unsqueeze_1847: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, 3);  unsqueeze_1846 = None
    mul_2187: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_1844);  sub_545 = unsqueeze_1844 = None
    sub_547: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_86, mul_2187);  where_86 = mul_2187 = None
    sub_548: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_547, unsqueeze_1841);  sub_547 = unsqueeze_1841 = None
    mul_2188: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_548, unsqueeze_1847);  sub_548 = unsqueeze_1847 = None
    mul_2189: "f32[432]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_343);  sum_175 = squeeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_160 = torch.ops.aten.convolution_backward.default(mul_2188, convolution_211, primals_447, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2188 = convolution_211 = primals_447 = None
    getitem_966: "f32[8, 432, 21, 21]" = convolution_backward_160[0]
    getitem_967: "f32[432, 432, 1, 1]" = convolution_backward_160[1];  convolution_backward_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_161 = torch.ops.aten.convolution_backward.default(getitem_966, relu_102, primals_446, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_966 = primals_446 = None
    getitem_969: "f32[8, 432, 21, 21]" = convolution_backward_161[0]
    getitem_970: "f32[432, 1, 3, 3]" = convolution_backward_161[1];  convolution_backward_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_87: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_102, 0)
    where_87: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_87, full_default, getitem_969);  getitem_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_18: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_36, add_554, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_261)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1129: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_37, max_pool2d_with_indices_backward_18);  slice_37 = max_pool2d_with_indices_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_176: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_36, [0, 2, 3])
    sub_549: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_210, unsqueeze_1850);  convolution_210 = unsqueeze_1850 = None
    mul_2190: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_36, sub_549)
    sum_177: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2190, [0, 2, 3]);  mul_2190 = None
    mul_2191: "f32[432]" = torch.ops.aten.mul.Tensor(sum_176, 0.0002834467120181406)
    unsqueeze_1851: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2191, 0);  mul_2191 = None
    unsqueeze_1852: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1851, 2);  unsqueeze_1851 = None
    unsqueeze_1853: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, 3);  unsqueeze_1852 = None
    mul_2192: "f32[432]" = torch.ops.aten.mul.Tensor(sum_177, 0.0002834467120181406)
    mul_2193: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
    mul_2194: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2192, mul_2193);  mul_2192 = mul_2193 = None
    unsqueeze_1854: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2194, 0);  mul_2194 = None
    unsqueeze_1855: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, 2);  unsqueeze_1854 = None
    unsqueeze_1856: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1855, 3);  unsqueeze_1855 = None
    mul_2195: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_340, primals_444);  primals_444 = None
    unsqueeze_1857: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2195, 0);  mul_2195 = None
    unsqueeze_1858: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1857, 2);  unsqueeze_1857 = None
    unsqueeze_1859: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, 3);  unsqueeze_1858 = None
    mul_2196: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_1856);  sub_549 = unsqueeze_1856 = None
    sub_551: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_36, mul_2196);  slice_36 = mul_2196 = None
    sub_552: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_551, unsqueeze_1853);  sub_551 = unsqueeze_1853 = None
    mul_2197: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_552, unsqueeze_1859);  sub_552 = unsqueeze_1859 = None
    mul_2198: "f32[432]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_340);  sum_177 = squeeze_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_162 = torch.ops.aten.convolution_backward.default(mul_2197, convolution_209, primals_443, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2197 = convolution_209 = primals_443 = None
    getitem_972: "f32[8, 432, 21, 21]" = convolution_backward_162[0]
    getitem_973: "f32[432, 432, 1, 1]" = convolution_backward_162[1];  convolution_backward_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_163 = torch.ops.aten.convolution_backward.default(getitem_972, relu_111, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_972 = primals_442 = None
    getitem_975: "f32[8, 432, 21, 21]" = convolution_backward_163[0]
    getitem_976: "f32[432, 1, 3, 3]" = convolution_backward_163[1];  convolution_backward_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_88: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_111, 0);  relu_111 = None
    where_88: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_88, full_default, getitem_975);  le_88 = getitem_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_178: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_553: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_208, unsqueeze_1862);  convolution_208 = unsqueeze_1862 = None
    mul_2199: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_88, sub_553)
    sum_179: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2199, [0, 2, 3]);  mul_2199 = None
    mul_2200: "f32[432]" = torch.ops.aten.mul.Tensor(sum_178, 0.0002834467120181406)
    unsqueeze_1863: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2200, 0);  mul_2200 = None
    unsqueeze_1864: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1863, 2);  unsqueeze_1863 = None
    unsqueeze_1865: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, 3);  unsqueeze_1864 = None
    mul_2201: "f32[432]" = torch.ops.aten.mul.Tensor(sum_179, 0.0002834467120181406)
    mul_2202: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
    mul_2203: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2201, mul_2202);  mul_2201 = mul_2202 = None
    unsqueeze_1866: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2203, 0);  mul_2203 = None
    unsqueeze_1867: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 2);  unsqueeze_1866 = None
    unsqueeze_1868: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1867, 3);  unsqueeze_1867 = None
    mul_2204: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_337, primals_440);  primals_440 = None
    unsqueeze_1869: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2204, 0);  mul_2204 = None
    unsqueeze_1870: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1869, 2);  unsqueeze_1869 = None
    unsqueeze_1871: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, 3);  unsqueeze_1870 = None
    mul_2205: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_1868);  sub_553 = unsqueeze_1868 = None
    sub_555: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_88, mul_2205);  where_88 = mul_2205 = None
    sub_556: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_555, unsqueeze_1865);  sub_555 = unsqueeze_1865 = None
    mul_2206: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_556, unsqueeze_1871);  sub_556 = unsqueeze_1871 = None
    mul_2207: "f32[432]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_337);  sum_179 = squeeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_164 = torch.ops.aten.convolution_backward.default(mul_2206, convolution_207, primals_439, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2206 = convolution_207 = primals_439 = None
    getitem_978: "f32[8, 432, 21, 21]" = convolution_backward_164[0]
    getitem_979: "f32[432, 432, 1, 1]" = convolution_backward_164[1];  convolution_backward_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_165 = torch.ops.aten.convolution_backward.default(getitem_978, relu_110, primals_438, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_978 = primals_438 = None
    getitem_981: "f32[8, 432, 21, 21]" = convolution_backward_165[0]
    getitem_982: "f32[432, 1, 3, 3]" = convolution_backward_165[1];  convolution_backward_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_89: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
    where_89: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_89, full_default, getitem_981);  le_89 = getitem_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1130: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_35, where_89);  slice_35 = where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_180: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1130, [0, 2, 3])
    sub_557: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_206, unsqueeze_1874);  convolution_206 = unsqueeze_1874 = None
    mul_2208: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1130, sub_557)
    sum_181: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2208, [0, 2, 3]);  mul_2208 = None
    mul_2209: "f32[432]" = torch.ops.aten.mul.Tensor(sum_180, 0.0002834467120181406)
    unsqueeze_1875: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2209, 0);  mul_2209 = None
    unsqueeze_1876: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1875, 2);  unsqueeze_1875 = None
    unsqueeze_1877: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, 3);  unsqueeze_1876 = None
    mul_2210: "f32[432]" = torch.ops.aten.mul.Tensor(sum_181, 0.0002834467120181406)
    mul_2211: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
    mul_2212: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2210, mul_2211);  mul_2210 = mul_2211 = None
    unsqueeze_1878: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2212, 0);  mul_2212 = None
    unsqueeze_1879: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 2);  unsqueeze_1878 = None
    unsqueeze_1880: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1879, 3);  unsqueeze_1879 = None
    mul_2213: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_334, primals_436);  primals_436 = None
    unsqueeze_1881: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2213, 0);  mul_2213 = None
    unsqueeze_1882: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1881, 2);  unsqueeze_1881 = None
    unsqueeze_1883: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, 3);  unsqueeze_1882 = None
    mul_2214: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_557, unsqueeze_1880);  sub_557 = unsqueeze_1880 = None
    sub_559: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1130, mul_2214);  mul_2214 = None
    sub_560: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_559, unsqueeze_1877);  sub_559 = None
    mul_2215: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_560, unsqueeze_1883);  sub_560 = unsqueeze_1883 = None
    mul_2216: "f32[432]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_334);  sum_181 = squeeze_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_166 = torch.ops.aten.convolution_backward.default(mul_2215, convolution_205, primals_435, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2215 = convolution_205 = primals_435 = None
    getitem_984: "f32[8, 432, 21, 21]" = convolution_backward_166[0]
    getitem_985: "f32[432, 432, 1, 1]" = convolution_backward_166[1];  convolution_backward_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_167 = torch.ops.aten.convolution_backward.default(getitem_984, relu_109, primals_434, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_984 = primals_434 = None
    getitem_987: "f32[8, 432, 21, 21]" = convolution_backward_167[0]
    getitem_988: "f32[432, 1, 3, 3]" = convolution_backward_167[1];  convolution_backward_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_90: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
    where_90: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_90, full_default, getitem_987);  le_90 = getitem_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_182: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_561: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_204, unsqueeze_1886);  convolution_204 = unsqueeze_1886 = None
    mul_2217: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_90, sub_561)
    sum_183: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2217, [0, 2, 3]);  mul_2217 = None
    mul_2218: "f32[432]" = torch.ops.aten.mul.Tensor(sum_182, 0.0002834467120181406)
    unsqueeze_1887: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2218, 0);  mul_2218 = None
    unsqueeze_1888: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1887, 2);  unsqueeze_1887 = None
    unsqueeze_1889: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, 3);  unsqueeze_1888 = None
    mul_2219: "f32[432]" = torch.ops.aten.mul.Tensor(sum_183, 0.0002834467120181406)
    mul_2220: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
    mul_2221: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2219, mul_2220);  mul_2219 = mul_2220 = None
    unsqueeze_1890: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2221, 0);  mul_2221 = None
    unsqueeze_1891: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 2);  unsqueeze_1890 = None
    unsqueeze_1892: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1891, 3);  unsqueeze_1891 = None
    mul_2222: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_331, primals_432);  primals_432 = None
    unsqueeze_1893: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2222, 0);  mul_2222 = None
    unsqueeze_1894: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1893, 2);  unsqueeze_1893 = None
    unsqueeze_1895: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, 3);  unsqueeze_1894 = None
    mul_2223: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_561, unsqueeze_1892);  sub_561 = unsqueeze_1892 = None
    sub_563: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_90, mul_2223);  where_90 = mul_2223 = None
    sub_564: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_563, unsqueeze_1889);  sub_563 = unsqueeze_1889 = None
    mul_2224: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_564, unsqueeze_1895);  sub_564 = unsqueeze_1895 = None
    mul_2225: "f32[432]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_331);  sum_183 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_168 = torch.ops.aten.convolution_backward.default(mul_2224, convolution_203, primals_431, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2224 = convolution_203 = primals_431 = None
    getitem_990: "f32[8, 432, 21, 21]" = convolution_backward_168[0]
    getitem_991: "f32[432, 432, 1, 1]" = convolution_backward_168[1];  convolution_backward_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_169 = torch.ops.aten.convolution_backward.default(getitem_990, relu_104, primals_430, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_990 = primals_430 = None
    getitem_993: "f32[8, 432, 21, 21]" = convolution_backward_169[0]
    getitem_994: "f32[432, 1, 3, 3]" = convolution_backward_169[1];  convolution_backward_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_91: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_104, 0)
    where_91: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_91, full_default, getitem_993);  getitem_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1131: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1129, where_91);  add_1129 = where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_565: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_202, unsqueeze_1898);  convolution_202 = unsqueeze_1898 = None
    mul_2226: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1130, sub_565)
    sum_185: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2226, [0, 2, 3]);  mul_2226 = None
    mul_2228: "f32[432]" = torch.ops.aten.mul.Tensor(sum_185, 0.0002834467120181406)
    mul_2229: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
    mul_2230: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2228, mul_2229);  mul_2228 = mul_2229 = None
    unsqueeze_1902: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2230, 0);  mul_2230 = None
    unsqueeze_1903: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 2);  unsqueeze_1902 = None
    unsqueeze_1904: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1903, 3);  unsqueeze_1903 = None
    mul_2231: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_328, primals_428);  primals_428 = None
    unsqueeze_1905: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2231, 0);  mul_2231 = None
    unsqueeze_1906: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1905, 2);  unsqueeze_1905 = None
    unsqueeze_1907: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, 3);  unsqueeze_1906 = None
    mul_2232: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_565, unsqueeze_1904);  sub_565 = unsqueeze_1904 = None
    sub_567: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1130, mul_2232);  add_1130 = mul_2232 = None
    sub_568: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_567, unsqueeze_1877);  sub_567 = unsqueeze_1877 = None
    mul_2233: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_568, unsqueeze_1907);  sub_568 = unsqueeze_1907 = None
    mul_2234: "f32[432]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_328);  sum_185 = squeeze_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_170 = torch.ops.aten.convolution_backward.default(mul_2233, convolution_201, primals_427, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2233 = convolution_201 = primals_427 = None
    getitem_996: "f32[8, 432, 21, 21]" = convolution_backward_170[0]
    getitem_997: "f32[432, 432, 1, 1]" = convolution_backward_170[1];  convolution_backward_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_171 = torch.ops.aten.convolution_backward.default(getitem_996, relu_107, primals_426, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_996 = primals_426 = None
    getitem_999: "f32[8, 432, 21, 21]" = convolution_backward_171[0]
    getitem_1000: "f32[432, 1, 5, 5]" = convolution_backward_171[1];  convolution_backward_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_92: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
    where_92: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_92, full_default, getitem_999);  le_92 = getitem_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_186: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_569: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_200, unsqueeze_1910);  convolution_200 = unsqueeze_1910 = None
    mul_2235: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_92, sub_569)
    sum_187: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2235, [0, 2, 3]);  mul_2235 = None
    mul_2236: "f32[432]" = torch.ops.aten.mul.Tensor(sum_186, 0.0002834467120181406)
    unsqueeze_1911: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2236, 0);  mul_2236 = None
    unsqueeze_1912: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1911, 2);  unsqueeze_1911 = None
    unsqueeze_1913: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, 3);  unsqueeze_1912 = None
    mul_2237: "f32[432]" = torch.ops.aten.mul.Tensor(sum_187, 0.0002834467120181406)
    mul_2238: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
    mul_2239: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2237, mul_2238);  mul_2237 = mul_2238 = None
    unsqueeze_1914: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2239, 0);  mul_2239 = None
    unsqueeze_1915: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, 2);  unsqueeze_1914 = None
    unsqueeze_1916: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1915, 3);  unsqueeze_1915 = None
    mul_2240: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_325, primals_424);  primals_424 = None
    unsqueeze_1917: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2240, 0);  mul_2240 = None
    unsqueeze_1918: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1917, 2);  unsqueeze_1917 = None
    unsqueeze_1919: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, 3);  unsqueeze_1918 = None
    mul_2241: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_569, unsqueeze_1916);  sub_569 = unsqueeze_1916 = None
    sub_571: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_92, mul_2241);  where_92 = mul_2241 = None
    sub_572: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_571, unsqueeze_1913);  sub_571 = unsqueeze_1913 = None
    mul_2242: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_572, unsqueeze_1919);  sub_572 = unsqueeze_1919 = None
    mul_2243: "f32[432]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_325);  sum_187 = squeeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_172 = torch.ops.aten.convolution_backward.default(mul_2242, convolution_199, primals_423, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2242 = convolution_199 = primals_423 = None
    getitem_1002: "f32[8, 432, 21, 21]" = convolution_backward_172[0]
    getitem_1003: "f32[432, 432, 1, 1]" = convolution_backward_172[1];  convolution_backward_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_173 = torch.ops.aten.convolution_backward.default(getitem_1002, relu_104, primals_422, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1002 = primals_422 = None
    getitem_1005: "f32[8, 432, 21, 21]" = convolution_backward_173[0]
    getitem_1006: "f32[432, 1, 5, 5]" = convolution_backward_173[1];  convolution_backward_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_93: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_91, full_default, getitem_1005);  getitem_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1132: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1131, where_93);  add_1131 = where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_19: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_34, add_554, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_261);  add_554 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1133: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1132, max_pool2d_with_indices_backward_19);  add_1132 = max_pool2d_with_indices_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_188: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_34, [0, 2, 3])
    sub_573: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_198, unsqueeze_1922);  convolution_198 = unsqueeze_1922 = None
    mul_2244: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_34, sub_573)
    sum_189: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2244, [0, 2, 3]);  mul_2244 = None
    mul_2245: "f32[432]" = torch.ops.aten.mul.Tensor(sum_188, 0.0002834467120181406)
    unsqueeze_1923: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2245, 0);  mul_2245 = None
    unsqueeze_1924: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1923, 2);  unsqueeze_1923 = None
    unsqueeze_1925: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, 3);  unsqueeze_1924 = None
    mul_2246: "f32[432]" = torch.ops.aten.mul.Tensor(sum_189, 0.0002834467120181406)
    mul_2247: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
    mul_2248: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2246, mul_2247);  mul_2246 = mul_2247 = None
    unsqueeze_1926: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2248, 0);  mul_2248 = None
    unsqueeze_1927: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, 2);  unsqueeze_1926 = None
    unsqueeze_1928: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1927, 3);  unsqueeze_1927 = None
    mul_2249: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_322, primals_420);  primals_420 = None
    unsqueeze_1929: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2249, 0);  mul_2249 = None
    unsqueeze_1930: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1929, 2);  unsqueeze_1929 = None
    unsqueeze_1931: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, 3);  unsqueeze_1930 = None
    mul_2250: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_573, unsqueeze_1928);  sub_573 = unsqueeze_1928 = None
    sub_575: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_34, mul_2250);  slice_34 = mul_2250 = None
    sub_576: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_575, unsqueeze_1925);  sub_575 = unsqueeze_1925 = None
    mul_2251: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_576, unsqueeze_1931);  sub_576 = unsqueeze_1931 = None
    mul_2252: "f32[432]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_322);  sum_189 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_174 = torch.ops.aten.convolution_backward.default(mul_2251, convolution_197, primals_419, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2251 = convolution_197 = primals_419 = None
    getitem_1008: "f32[8, 432, 21, 21]" = convolution_backward_174[0]
    getitem_1009: "f32[432, 432, 1, 1]" = convolution_backward_174[1];  convolution_backward_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_175 = torch.ops.aten.convolution_backward.default(getitem_1008, relu_105, primals_418, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1008 = primals_418 = None
    getitem_1011: "f32[8, 432, 21, 21]" = convolution_backward_175[0]
    getitem_1012: "f32[432, 1, 7, 7]" = convolution_backward_175[1];  convolution_backward_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_94: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
    where_94: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_94, full_default, getitem_1011);  le_94 = getitem_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_190: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_577: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_196, unsqueeze_1934);  convolution_196 = unsqueeze_1934 = None
    mul_2253: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_94, sub_577)
    sum_191: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2253, [0, 2, 3]);  mul_2253 = None
    mul_2254: "f32[432]" = torch.ops.aten.mul.Tensor(sum_190, 0.0002834467120181406)
    unsqueeze_1935: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2254, 0);  mul_2254 = None
    unsqueeze_1936: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1935, 2);  unsqueeze_1935 = None
    unsqueeze_1937: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1936, 3);  unsqueeze_1936 = None
    mul_2255: "f32[432]" = torch.ops.aten.mul.Tensor(sum_191, 0.0002834467120181406)
    mul_2256: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
    mul_2257: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2255, mul_2256);  mul_2255 = mul_2256 = None
    unsqueeze_1938: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2257, 0);  mul_2257 = None
    unsqueeze_1939: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, 2);  unsqueeze_1938 = None
    unsqueeze_1940: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1939, 3);  unsqueeze_1939 = None
    mul_2258: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_319, primals_416);  primals_416 = None
    unsqueeze_1941: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2258, 0);  mul_2258 = None
    unsqueeze_1942: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1941, 2);  unsqueeze_1941 = None
    unsqueeze_1943: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1942, 3);  unsqueeze_1942 = None
    mul_2259: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_577, unsqueeze_1940);  sub_577 = unsqueeze_1940 = None
    sub_579: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_94, mul_2259);  where_94 = mul_2259 = None
    sub_580: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_579, unsqueeze_1937);  sub_579 = unsqueeze_1937 = None
    mul_2260: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_580, unsqueeze_1943);  sub_580 = unsqueeze_1943 = None
    mul_2261: "f32[432]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_319);  sum_191 = squeeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_176 = torch.ops.aten.convolution_backward.default(mul_2260, convolution_195, primals_415, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2260 = convolution_195 = primals_415 = None
    getitem_1014: "f32[8, 432, 21, 21]" = convolution_backward_176[0]
    getitem_1015: "f32[432, 432, 1, 1]" = convolution_backward_176[1];  convolution_backward_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_177 = torch.ops.aten.convolution_backward.default(getitem_1014, relu_104, primals_414, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1014 = relu_104 = primals_414 = None
    getitem_1017: "f32[8, 432, 21, 21]" = convolution_backward_177[0]
    getitem_1018: "f32[432, 1, 7, 7]" = convolution_backward_177[1];  convolution_backward_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_95: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_91, full_default, getitem_1017);  le_91 = getitem_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1134: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1133, where_95);  add_1133 = where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_20: "f32[8, 432, 21, 21]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_33, add_549, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_255);  add_549 = getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1135: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(where_87, max_pool2d_with_indices_backward_20);  where_87 = max_pool2d_with_indices_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_192: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_33, [0, 2, 3])
    sub_581: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_194, unsqueeze_1946);  convolution_194 = unsqueeze_1946 = None
    mul_2262: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_33, sub_581)
    sum_193: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2262, [0, 2, 3]);  mul_2262 = None
    mul_2263: "f32[432]" = torch.ops.aten.mul.Tensor(sum_192, 0.0002834467120181406)
    unsqueeze_1947: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2263, 0);  mul_2263 = None
    unsqueeze_1948: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1947, 2);  unsqueeze_1947 = None
    unsqueeze_1949: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1948, 3);  unsqueeze_1948 = None
    mul_2264: "f32[432]" = torch.ops.aten.mul.Tensor(sum_193, 0.0002834467120181406)
    mul_2265: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
    mul_2266: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2264, mul_2265);  mul_2264 = mul_2265 = None
    unsqueeze_1950: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2266, 0);  mul_2266 = None
    unsqueeze_1951: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 2);  unsqueeze_1950 = None
    unsqueeze_1952: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1951, 3);  unsqueeze_1951 = None
    mul_2267: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_316, primals_412);  primals_412 = None
    unsqueeze_1953: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2267, 0);  mul_2267 = None
    unsqueeze_1954: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1953, 2);  unsqueeze_1953 = None
    unsqueeze_1955: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1954, 3);  unsqueeze_1954 = None
    mul_2268: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_581, unsqueeze_1952);  sub_581 = unsqueeze_1952 = None
    sub_583: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_33, mul_2268);  slice_33 = mul_2268 = None
    sub_584: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_583, unsqueeze_1949);  sub_583 = unsqueeze_1949 = None
    mul_2269: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_584, unsqueeze_1955);  sub_584 = unsqueeze_1955 = None
    mul_2270: "f32[432]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_316);  sum_193 = squeeze_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_178 = torch.ops.aten.convolution_backward.default(mul_2269, convolution_193, primals_411, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2269 = convolution_193 = primals_411 = None
    getitem_1020: "f32[8, 432, 21, 21]" = convolution_backward_178[0]
    getitem_1021: "f32[432, 432, 1, 1]" = convolution_backward_178[1];  convolution_backward_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_179 = torch.ops.aten.convolution_backward.default(getitem_1020, relu_103, primals_410, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1020 = primals_410 = None
    getitem_1023: "f32[8, 432, 21, 21]" = convolution_backward_179[0]
    getitem_1024: "f32[432, 1, 5, 5]" = convolution_backward_179[1];  convolution_backward_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_96: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
    where_96: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_96, full_default, getitem_1023);  le_96 = getitem_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_194: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_585: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_192, unsqueeze_1958);  convolution_192 = unsqueeze_1958 = None
    mul_2271: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_96, sub_585)
    sum_195: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2271, [0, 2, 3]);  mul_2271 = None
    mul_2272: "f32[432]" = torch.ops.aten.mul.Tensor(sum_194, 0.0002834467120181406)
    unsqueeze_1959: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2272, 0);  mul_2272 = None
    unsqueeze_1960: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1959, 2);  unsqueeze_1959 = None
    unsqueeze_1961: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1960, 3);  unsqueeze_1960 = None
    mul_2273: "f32[432]" = torch.ops.aten.mul.Tensor(sum_195, 0.0002834467120181406)
    mul_2274: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_2275: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2273, mul_2274);  mul_2273 = mul_2274 = None
    unsqueeze_1962: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2275, 0);  mul_2275 = None
    unsqueeze_1963: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 2);  unsqueeze_1962 = None
    unsqueeze_1964: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1963, 3);  unsqueeze_1963 = None
    mul_2276: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_408);  primals_408 = None
    unsqueeze_1965: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2276, 0);  mul_2276 = None
    unsqueeze_1966: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1965, 2);  unsqueeze_1965 = None
    unsqueeze_1967: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1966, 3);  unsqueeze_1966 = None
    mul_2277: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_585, unsqueeze_1964);  sub_585 = unsqueeze_1964 = None
    sub_587: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_96, mul_2277);  where_96 = mul_2277 = None
    sub_588: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_587, unsqueeze_1961);  sub_587 = unsqueeze_1961 = None
    mul_2278: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_588, unsqueeze_1967);  sub_588 = unsqueeze_1967 = None
    mul_2279: "f32[432]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_313);  sum_195 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_180 = torch.ops.aten.convolution_backward.default(mul_2278, convolution_191, primals_407, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2278 = convolution_191 = primals_407 = None
    getitem_1026: "f32[8, 432, 21, 21]" = convolution_backward_180[0]
    getitem_1027: "f32[432, 432, 1, 1]" = convolution_backward_180[1];  convolution_backward_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_181 = torch.ops.aten.convolution_backward.default(getitem_1026, relu_102, primals_406, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1026 = relu_102 = primals_406 = None
    getitem_1029: "f32[8, 432, 21, 21]" = convolution_backward_181[0]
    getitem_1030: "f32[432, 1, 5, 5]" = convolution_backward_181[1];  convolution_backward_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_97: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_87, full_default, getitem_1029);  le_87 = getitem_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1136: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(add_1135, where_97);  add_1135 = where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_196: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1134, [0, 2, 3])
    sub_589: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_190, unsqueeze_1970);  convolution_190 = unsqueeze_1970 = None
    mul_2280: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1134, sub_589)
    sum_197: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2280, [0, 2, 3]);  mul_2280 = None
    mul_2281: "f32[432]" = torch.ops.aten.mul.Tensor(sum_196, 0.0002834467120181406)
    unsqueeze_1971: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2281, 0);  mul_2281 = None
    unsqueeze_1972: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1971, 2);  unsqueeze_1971 = None
    unsqueeze_1973: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1972, 3);  unsqueeze_1972 = None
    mul_2282: "f32[432]" = torch.ops.aten.mul.Tensor(sum_197, 0.0002834467120181406)
    mul_2283: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_2284: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2282, mul_2283);  mul_2282 = mul_2283 = None
    unsqueeze_1974: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2284, 0);  mul_2284 = None
    unsqueeze_1975: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 2);  unsqueeze_1974 = None
    unsqueeze_1976: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1975, 3);  unsqueeze_1975 = None
    mul_2285: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_404);  primals_404 = None
    unsqueeze_1977: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2285, 0);  mul_2285 = None
    unsqueeze_1978: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1977, 2);  unsqueeze_1977 = None
    unsqueeze_1979: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1978, 3);  unsqueeze_1978 = None
    mul_2286: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_1976);  sub_589 = unsqueeze_1976 = None
    sub_591: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1134, mul_2286);  add_1134 = mul_2286 = None
    sub_592: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_591, unsqueeze_1973);  sub_591 = unsqueeze_1973 = None
    mul_2287: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_592, unsqueeze_1979);  sub_592 = unsqueeze_1979 = None
    mul_2288: "f32[432]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_310);  sum_197 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_182 = torch.ops.aten.convolution_backward.default(mul_2287, relu_101, primals_403, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2287 = relu_101 = primals_403 = None
    getitem_1032: "f32[8, 2160, 21, 21]" = convolution_backward_182[0]
    getitem_1033: "f32[432, 2160, 1, 1]" = convolution_backward_182[1];  convolution_backward_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_98: "f32[8, 2160, 21, 21]" = torch.ops.aten.where.self(le_85, full_default, getitem_1032);  le_85 = getitem_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1137: "f32[8, 2160, 21, 21]" = torch.ops.aten.add.Tensor(where_85, where_98);  where_85 = where_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    sum_198: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1136, [0, 2, 3])
    sub_593: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_1982);  cat_9 = unsqueeze_1982 = None
    mul_2289: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1136, sub_593)
    sum_199: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2289, [0, 2, 3]);  mul_2289 = None
    mul_2290: "f32[432]" = torch.ops.aten.mul.Tensor(sum_198, 0.0002834467120181406)
    unsqueeze_1983: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2290, 0);  mul_2290 = None
    unsqueeze_1984: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1983, 2);  unsqueeze_1983 = None
    unsqueeze_1985: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1984, 3);  unsqueeze_1984 = None
    mul_2291: "f32[432]" = torch.ops.aten.mul.Tensor(sum_199, 0.0002834467120181406)
    mul_2292: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_2293: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2291, mul_2292);  mul_2291 = mul_2292 = None
    unsqueeze_1986: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2293, 0);  mul_2293 = None
    unsqueeze_1987: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 2);  unsqueeze_1986 = None
    unsqueeze_1988: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1987, 3);  unsqueeze_1987 = None
    mul_2294: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_401);  primals_401 = None
    unsqueeze_1989: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2294, 0);  mul_2294 = None
    unsqueeze_1990: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1989, 2);  unsqueeze_1989 = None
    unsqueeze_1991: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1990, 3);  unsqueeze_1990 = None
    mul_2295: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_1988);  sub_593 = unsqueeze_1988 = None
    sub_595: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1136, mul_2295);  add_1136 = mul_2295 = None
    sub_596: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_595, unsqueeze_1985);  sub_595 = unsqueeze_1985 = None
    mul_2296: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_596, unsqueeze_1991);  sub_596 = unsqueeze_1991 = None
    mul_2297: "f32[432]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_307);  sum_199 = squeeze_307 = None
    slice_38: "f32[8, 216, 21, 21]" = torch.ops.aten.slice.Tensor(mul_2296, 1, 0, 216)
    slice_39: "f32[8, 216, 21, 21]" = torch.ops.aten.slice.Tensor(mul_2296, 1, 216, 432);  mul_2296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    convolution_backward_183 = torch.ops.aten.convolution_backward.default(slice_39, avg_pool2d_5, primals_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_39 = avg_pool2d_5 = primals_400 = None
    getitem_1035: "f32[8, 1080, 21, 21]" = convolution_backward_183[0]
    getitem_1036: "f32[216, 1080, 1, 1]" = convolution_backward_183[1];  convolution_backward_183 = None
    avg_pool2d_backward_2: "f32[8, 1080, 42, 42]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1035, constant_pad_nd_29, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1035 = constant_pad_nd_29 = None
    constant_pad_nd_50: "f32[8, 1080, 42, 42]" = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_2, [1, -1, 1, -1]);  avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    convolution_backward_184 = torch.ops.aten.convolution_backward.default(slice_38, avg_pool2d_4, primals_399, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_38 = avg_pool2d_4 = primals_399 = None
    getitem_1038: "f32[8, 1080, 21, 21]" = convolution_backward_184[0]
    getitem_1039: "f32[216, 1080, 1, 1]" = convolution_backward_184[1];  convolution_backward_184 = None
    avg_pool2d_backward_3: "f32[8, 1080, 42, 42]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1038, relu_86, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    add_1138: "f32[8, 1080, 42, 42]" = torch.ops.aten.add.Tensor(constant_pad_nd_50, avg_pool2d_backward_3);  constant_pad_nd_50 = avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    le_99: "b8[8, 1080, 42, 42]" = torch.ops.aten.le.Scalar(relu_86, 0)
    where_99: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_99, full_default, add_1138);  add_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_40: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1137, 1, 0, 432)
    slice_41: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1137, 1, 432, 864)
    slice_42: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1137, 1, 864, 1296)
    slice_43: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1137, 1, 1296, 1728)
    slice_44: "f32[8, 432, 21, 21]" = torch.ops.aten.slice.Tensor(add_1137, 1, 1728, 2160);  add_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_200: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_44, [0, 2, 3])
    sub_597: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_187, unsqueeze_1994);  convolution_187 = unsqueeze_1994 = None
    mul_2298: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_44, sub_597)
    sum_201: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2298, [0, 2, 3]);  mul_2298 = None
    mul_2299: "f32[432]" = torch.ops.aten.mul.Tensor(sum_200, 0.0002834467120181406)
    unsqueeze_1995: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2299, 0);  mul_2299 = None
    unsqueeze_1996: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1995, 2);  unsqueeze_1995 = None
    unsqueeze_1997: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1996, 3);  unsqueeze_1996 = None
    mul_2300: "f32[432]" = torch.ops.aten.mul.Tensor(sum_201, 0.0002834467120181406)
    mul_2301: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_2302: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2300, mul_2301);  mul_2300 = mul_2301 = None
    unsqueeze_1998: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2302, 0);  mul_2302 = None
    unsqueeze_1999: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, 2);  unsqueeze_1998 = None
    unsqueeze_2000: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1999, 3);  unsqueeze_1999 = None
    mul_2303: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_397);  primals_397 = None
    unsqueeze_2001: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2303, 0);  mul_2303 = None
    unsqueeze_2002: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2001, 2);  unsqueeze_2001 = None
    unsqueeze_2003: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2002, 3);  unsqueeze_2002 = None
    mul_2304: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_2000);  sub_597 = unsqueeze_2000 = None
    sub_599: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_44, mul_2304);  mul_2304 = None
    sub_600: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_599, unsqueeze_1997);  sub_599 = None
    mul_2305: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_600, unsqueeze_2003);  sub_600 = unsqueeze_2003 = None
    mul_2306: "f32[432]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_304);  sum_201 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_185 = torch.ops.aten.convolution_backward.default(mul_2305, constant_pad_nd_28, primals_20, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2305 = constant_pad_nd_28 = primals_20 = None
    getitem_1041: "f32[8, 432, 42, 42]" = convolution_backward_185[0]
    getitem_1042: "f32[432, 432, 1, 1]" = convolution_backward_185[1];  convolution_backward_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_100: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_100, full_default, getitem_1041);  getitem_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_601: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_186, unsqueeze_2006);  convolution_186 = unsqueeze_2006 = None
    mul_2307: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_44, sub_601)
    sum_203: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2307, [0, 2, 3]);  mul_2307 = None
    mul_2309: "f32[432]" = torch.ops.aten.mul.Tensor(sum_203, 0.0002834467120181406)
    mul_2310: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_2311: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2309, mul_2310);  mul_2309 = mul_2310 = None
    unsqueeze_2010: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2311, 0);  mul_2311 = None
    unsqueeze_2011: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 2);  unsqueeze_2010 = None
    unsqueeze_2012: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2011, 3);  unsqueeze_2011 = None
    mul_2312: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_395);  primals_395 = None
    unsqueeze_2013: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2312, 0);  mul_2312 = None
    unsqueeze_2014: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2013, 2);  unsqueeze_2013 = None
    unsqueeze_2015: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2014, 3);  unsqueeze_2014 = None
    mul_2313: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_2012);  sub_601 = unsqueeze_2012 = None
    sub_603: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_44, mul_2313);  slice_44 = mul_2313 = None
    sub_604: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_603, unsqueeze_1997);  sub_603 = unsqueeze_1997 = None
    mul_2314: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_604, unsqueeze_2015);  sub_604 = unsqueeze_2015 = None
    mul_2315: "f32[432]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_301);  sum_203 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_186 = torch.ops.aten.convolution_backward.default(mul_2314, convolution_185, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2314 = convolution_185 = primals_394 = None
    getitem_1044: "f32[8, 432, 21, 21]" = convolution_backward_186[0]
    getitem_1045: "f32[432, 432, 1, 1]" = convolution_backward_186[1];  convolution_backward_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_187 = torch.ops.aten.convolution_backward.default(getitem_1044, relu_98, primals_393, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1044 = primals_393 = None
    getitem_1047: "f32[8, 432, 21, 21]" = convolution_backward_187[0]
    getitem_1048: "f32[432, 1, 3, 3]" = convolution_backward_187[1];  convolution_backward_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_101: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
    where_101: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_101, full_default, getitem_1047);  le_101 = getitem_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_204: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_605: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_184, unsqueeze_2018);  convolution_184 = unsqueeze_2018 = None
    mul_2316: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_101, sub_605)
    sum_205: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2316, [0, 2, 3]);  mul_2316 = None
    mul_2317: "f32[432]" = torch.ops.aten.mul.Tensor(sum_204, 0.0002834467120181406)
    unsqueeze_2019: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2317, 0);  mul_2317 = None
    unsqueeze_2020: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2019, 2);  unsqueeze_2019 = None
    unsqueeze_2021: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2020, 3);  unsqueeze_2020 = None
    mul_2318: "f32[432]" = torch.ops.aten.mul.Tensor(sum_205, 0.0002834467120181406)
    mul_2319: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_2320: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2318, mul_2319);  mul_2318 = mul_2319 = None
    unsqueeze_2022: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2320, 0);  mul_2320 = None
    unsqueeze_2023: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 2);  unsqueeze_2022 = None
    unsqueeze_2024: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2023, 3);  unsqueeze_2023 = None
    mul_2321: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_391);  primals_391 = None
    unsqueeze_2025: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2321, 0);  mul_2321 = None
    unsqueeze_2026: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2025, 2);  unsqueeze_2025 = None
    unsqueeze_2027: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2026, 3);  unsqueeze_2026 = None
    mul_2322: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_605, unsqueeze_2024);  sub_605 = unsqueeze_2024 = None
    sub_607: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_101, mul_2322);  where_101 = mul_2322 = None
    sub_608: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_607, unsqueeze_2021);  sub_607 = unsqueeze_2021 = None
    mul_2323: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_608, unsqueeze_2027);  sub_608 = unsqueeze_2027 = None
    mul_2324: "f32[432]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_298);  sum_205 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_188 = torch.ops.aten.convolution_backward.default(mul_2323, convolution_183, primals_390, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2323 = convolution_183 = primals_390 = None
    getitem_1050: "f32[8, 432, 21, 21]" = convolution_backward_188[0]
    getitem_1051: "f32[432, 432, 1, 1]" = convolution_backward_188[1];  convolution_backward_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_189 = torch.ops.aten.convolution_backward.default(getitem_1050, constant_pad_nd_27, primals_19, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1050 = constant_pad_nd_27 = primals_19 = None
    getitem_1053: "f32[8, 432, 43, 43]" = convolution_backward_189[0]
    getitem_1054: "f32[432, 1, 3, 3]" = convolution_backward_189[1];  convolution_backward_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_52: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(getitem_1053, [0, -1, 0, -1]);  getitem_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_102: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_102, full_default, constant_pad_nd_52);  constant_pad_nd_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_21: "f32[8, 432, 43, 43]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_43, constant_pad_nd_23, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_225)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_53: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_21, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1139: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(where_100, constant_pad_nd_53);  where_100 = constant_pad_nd_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_206: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_43, [0, 2, 3])
    sub_609: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_182, unsqueeze_2030);  convolution_182 = unsqueeze_2030 = None
    mul_2325: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_43, sub_609)
    sum_207: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2325, [0, 2, 3]);  mul_2325 = None
    mul_2326: "f32[432]" = torch.ops.aten.mul.Tensor(sum_206, 0.0002834467120181406)
    unsqueeze_2031: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2326, 0);  mul_2326 = None
    unsqueeze_2032: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2031, 2);  unsqueeze_2031 = None
    unsqueeze_2033: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2032, 3);  unsqueeze_2032 = None
    mul_2327: "f32[432]" = torch.ops.aten.mul.Tensor(sum_207, 0.0002834467120181406)
    mul_2328: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_2329: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2327, mul_2328);  mul_2327 = mul_2328 = None
    unsqueeze_2034: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2329, 0);  mul_2329 = None
    unsqueeze_2035: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 2);  unsqueeze_2034 = None
    unsqueeze_2036: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2035, 3);  unsqueeze_2035 = None
    mul_2330: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_388);  primals_388 = None
    unsqueeze_2037: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2330, 0);  mul_2330 = None
    unsqueeze_2038: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2037, 2);  unsqueeze_2037 = None
    unsqueeze_2039: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2038, 3);  unsqueeze_2038 = None
    mul_2331: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_609, unsqueeze_2036);  sub_609 = unsqueeze_2036 = None
    sub_611: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_43, mul_2331);  slice_43 = mul_2331 = None
    sub_612: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_611, unsqueeze_2033);  sub_611 = unsqueeze_2033 = None
    mul_2332: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_612, unsqueeze_2039);  sub_612 = unsqueeze_2039 = None
    mul_2333: "f32[432]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_295);  sum_207 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_190 = torch.ops.aten.convolution_backward.default(mul_2332, convolution_181, primals_387, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2332 = convolution_181 = primals_387 = None
    getitem_1056: "f32[8, 432, 21, 21]" = convolution_backward_190[0]
    getitem_1057: "f32[432, 432, 1, 1]" = convolution_backward_190[1];  convolution_backward_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_191 = torch.ops.aten.convolution_backward.default(getitem_1056, relu_96, primals_386, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1056 = primals_386 = None
    getitem_1059: "f32[8, 432, 21, 21]" = convolution_backward_191[0]
    getitem_1060: "f32[432, 1, 3, 3]" = convolution_backward_191[1];  convolution_backward_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_103: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
    where_103: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_103, full_default, getitem_1059);  le_103 = getitem_1059 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_208: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_613: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_180, unsqueeze_2042);  convolution_180 = unsqueeze_2042 = None
    mul_2334: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_103, sub_613)
    sum_209: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2334, [0, 2, 3]);  mul_2334 = None
    mul_2335: "f32[432]" = torch.ops.aten.mul.Tensor(sum_208, 0.0002834467120181406)
    unsqueeze_2043: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2335, 0);  mul_2335 = None
    unsqueeze_2044: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2043, 2);  unsqueeze_2043 = None
    unsqueeze_2045: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2044, 3);  unsqueeze_2044 = None
    mul_2336: "f32[432]" = torch.ops.aten.mul.Tensor(sum_209, 0.0002834467120181406)
    mul_2337: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_2338: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2336, mul_2337);  mul_2336 = mul_2337 = None
    unsqueeze_2046: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2338, 0);  mul_2338 = None
    unsqueeze_2047: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, 2);  unsqueeze_2046 = None
    unsqueeze_2048: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2047, 3);  unsqueeze_2047 = None
    mul_2339: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_384);  primals_384 = None
    unsqueeze_2049: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2339, 0);  mul_2339 = None
    unsqueeze_2050: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2049, 2);  unsqueeze_2049 = None
    unsqueeze_2051: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2050, 3);  unsqueeze_2050 = None
    mul_2340: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_613, unsqueeze_2048);  sub_613 = unsqueeze_2048 = None
    sub_615: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_103, mul_2340);  where_103 = mul_2340 = None
    sub_616: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_615, unsqueeze_2045);  sub_615 = unsqueeze_2045 = None
    mul_2341: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_616, unsqueeze_2051);  sub_616 = unsqueeze_2051 = None
    mul_2342: "f32[432]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_292);  sum_209 = squeeze_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_192 = torch.ops.aten.convolution_backward.default(mul_2341, convolution_179, primals_383, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2341 = convolution_179 = primals_383 = None
    getitem_1062: "f32[8, 432, 21, 21]" = convolution_backward_192[0]
    getitem_1063: "f32[432, 432, 1, 1]" = convolution_backward_192[1];  convolution_backward_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_193 = torch.ops.aten.convolution_backward.default(getitem_1062, relu_95, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1062 = primals_382 = None
    getitem_1065: "f32[8, 432, 21, 21]" = convolution_backward_193[0]
    getitem_1066: "f32[432, 1, 3, 3]" = convolution_backward_193[1];  convolution_backward_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_104: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
    where_104: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_104, full_default, getitem_1065);  le_104 = getitem_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1140: "f32[8, 432, 21, 21]" = torch.ops.aten.add.Tensor(slice_42, where_104);  slice_42 = where_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_210: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1140, [0, 2, 3])
    sub_617: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_178, unsqueeze_2054);  convolution_178 = unsqueeze_2054 = None
    mul_2343: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1140, sub_617)
    sum_211: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2343, [0, 2, 3]);  mul_2343 = None
    mul_2344: "f32[432]" = torch.ops.aten.mul.Tensor(sum_210, 0.0002834467120181406)
    unsqueeze_2055: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2344, 0);  mul_2344 = None
    unsqueeze_2056: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2055, 2);  unsqueeze_2055 = None
    unsqueeze_2057: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2056, 3);  unsqueeze_2056 = None
    mul_2345: "f32[432]" = torch.ops.aten.mul.Tensor(sum_211, 0.0002834467120181406)
    mul_2346: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_2347: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2345, mul_2346);  mul_2345 = mul_2346 = None
    unsqueeze_2058: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2347, 0);  mul_2347 = None
    unsqueeze_2059: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 2);  unsqueeze_2058 = None
    unsqueeze_2060: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2059, 3);  unsqueeze_2059 = None
    mul_2348: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_380);  primals_380 = None
    unsqueeze_2061: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2348, 0);  mul_2348 = None
    unsqueeze_2062: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2061, 2);  unsqueeze_2061 = None
    unsqueeze_2063: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2062, 3);  unsqueeze_2062 = None
    mul_2349: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_617, unsqueeze_2060);  sub_617 = unsqueeze_2060 = None
    sub_619: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1140, mul_2349);  mul_2349 = None
    sub_620: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_619, unsqueeze_2057);  sub_619 = None
    mul_2350: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_620, unsqueeze_2063);  sub_620 = unsqueeze_2063 = None
    mul_2351: "f32[432]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_289);  sum_211 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_194 = torch.ops.aten.convolution_backward.default(mul_2350, convolution_177, primals_379, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2350 = convolution_177 = primals_379 = None
    getitem_1068: "f32[8, 432, 21, 21]" = convolution_backward_194[0]
    getitem_1069: "f32[432, 432, 1, 1]" = convolution_backward_194[1];  convolution_backward_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_195 = torch.ops.aten.convolution_backward.default(getitem_1068, relu_94, primals_378, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1068 = primals_378 = None
    getitem_1071: "f32[8, 432, 21, 21]" = convolution_backward_195[0]
    getitem_1072: "f32[432, 1, 3, 3]" = convolution_backward_195[1];  convolution_backward_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_105: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
    where_105: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_105, full_default, getitem_1071);  le_105 = getitem_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_212: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_621: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_176, unsqueeze_2066);  convolution_176 = unsqueeze_2066 = None
    mul_2352: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_105, sub_621)
    sum_213: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2352, [0, 2, 3]);  mul_2352 = None
    mul_2353: "f32[432]" = torch.ops.aten.mul.Tensor(sum_212, 0.0002834467120181406)
    unsqueeze_2067: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2353, 0);  mul_2353 = None
    unsqueeze_2068: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2067, 2);  unsqueeze_2067 = None
    unsqueeze_2069: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2068, 3);  unsqueeze_2068 = None
    mul_2354: "f32[432]" = torch.ops.aten.mul.Tensor(sum_213, 0.0002834467120181406)
    mul_2355: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_2356: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2354, mul_2355);  mul_2354 = mul_2355 = None
    unsqueeze_2070: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2356, 0);  mul_2356 = None
    unsqueeze_2071: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, 2);  unsqueeze_2070 = None
    unsqueeze_2072: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2071, 3);  unsqueeze_2071 = None
    mul_2357: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_376);  primals_376 = None
    unsqueeze_2073: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2357, 0);  mul_2357 = None
    unsqueeze_2074: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2073, 2);  unsqueeze_2073 = None
    unsqueeze_2075: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2074, 3);  unsqueeze_2074 = None
    mul_2358: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_621, unsqueeze_2072);  sub_621 = unsqueeze_2072 = None
    sub_623: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_105, mul_2358);  where_105 = mul_2358 = None
    sub_624: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_623, unsqueeze_2069);  sub_623 = unsqueeze_2069 = None
    mul_2359: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_624, unsqueeze_2075);  sub_624 = unsqueeze_2075 = None
    mul_2360: "f32[432]" = torch.ops.aten.mul.Tensor(sum_213, squeeze_286);  sum_213 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_196 = torch.ops.aten.convolution_backward.default(mul_2359, convolution_175, primals_375, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2359 = convolution_175 = primals_375 = None
    getitem_1074: "f32[8, 432, 21, 21]" = convolution_backward_196[0]
    getitem_1075: "f32[432, 432, 1, 1]" = convolution_backward_196[1];  convolution_backward_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_197 = torch.ops.aten.convolution_backward.default(getitem_1074, constant_pad_nd_25, primals_18, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1074 = constant_pad_nd_25 = primals_18 = None
    getitem_1077: "f32[8, 432, 43, 43]" = convolution_backward_197[0]
    getitem_1078: "f32[432, 1, 3, 3]" = convolution_backward_197[1];  convolution_backward_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_54: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(getitem_1077, [0, -1, 0, -1]);  getitem_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_106: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_100, full_default, constant_pad_nd_54);  constant_pad_nd_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1141: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(add_1139, where_106);  add_1139 = where_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_625: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_174, unsqueeze_2078);  convolution_174 = unsqueeze_2078 = None
    mul_2361: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(add_1140, sub_625)
    sum_215: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2361, [0, 2, 3]);  mul_2361 = None
    mul_2363: "f32[432]" = torch.ops.aten.mul.Tensor(sum_215, 0.0002834467120181406)
    mul_2364: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_2365: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2363, mul_2364);  mul_2363 = mul_2364 = None
    unsqueeze_2082: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2365, 0);  mul_2365 = None
    unsqueeze_2083: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, 2);  unsqueeze_2082 = None
    unsqueeze_2084: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2083, 3);  unsqueeze_2083 = None
    mul_2366: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_373);  primals_373 = None
    unsqueeze_2085: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2366, 0);  mul_2366 = None
    unsqueeze_2086: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2085, 2);  unsqueeze_2085 = None
    unsqueeze_2087: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2086, 3);  unsqueeze_2086 = None
    mul_2367: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_625, unsqueeze_2084);  sub_625 = unsqueeze_2084 = None
    sub_627: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(add_1140, mul_2367);  add_1140 = mul_2367 = None
    sub_628: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_627, unsqueeze_2057);  sub_627 = unsqueeze_2057 = None
    mul_2368: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_628, unsqueeze_2087);  sub_628 = unsqueeze_2087 = None
    mul_2369: "f32[432]" = torch.ops.aten.mul.Tensor(sum_215, squeeze_283);  sum_215 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_198 = torch.ops.aten.convolution_backward.default(mul_2368, convolution_173, primals_372, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2368 = convolution_173 = primals_372 = None
    getitem_1080: "f32[8, 432, 21, 21]" = convolution_backward_198[0]
    getitem_1081: "f32[432, 432, 1, 1]" = convolution_backward_198[1];  convolution_backward_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_199 = torch.ops.aten.convolution_backward.default(getitem_1080, relu_92, primals_371, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1080 = primals_371 = None
    getitem_1083: "f32[8, 432, 21, 21]" = convolution_backward_199[0]
    getitem_1084: "f32[432, 1, 5, 5]" = convolution_backward_199[1];  convolution_backward_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_107: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
    where_107: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_107, full_default, getitem_1083);  le_107 = getitem_1083 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_216: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_629: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_172, unsqueeze_2090);  convolution_172 = unsqueeze_2090 = None
    mul_2370: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_107, sub_629)
    sum_217: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2370, [0, 2, 3]);  mul_2370 = None
    mul_2371: "f32[432]" = torch.ops.aten.mul.Tensor(sum_216, 0.0002834467120181406)
    unsqueeze_2091: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2371, 0);  mul_2371 = None
    unsqueeze_2092: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2091, 2);  unsqueeze_2091 = None
    unsqueeze_2093: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2092, 3);  unsqueeze_2092 = None
    mul_2372: "f32[432]" = torch.ops.aten.mul.Tensor(sum_217, 0.0002834467120181406)
    mul_2373: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_2374: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2372, mul_2373);  mul_2372 = mul_2373 = None
    unsqueeze_2094: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2374, 0);  mul_2374 = None
    unsqueeze_2095: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, 2);  unsqueeze_2094 = None
    unsqueeze_2096: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2095, 3);  unsqueeze_2095 = None
    mul_2375: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_369);  primals_369 = None
    unsqueeze_2097: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2375, 0);  mul_2375 = None
    unsqueeze_2098: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2097, 2);  unsqueeze_2097 = None
    unsqueeze_2099: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2098, 3);  unsqueeze_2098 = None
    mul_2376: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_629, unsqueeze_2096);  sub_629 = unsqueeze_2096 = None
    sub_631: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_107, mul_2376);  where_107 = mul_2376 = None
    sub_632: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_631, unsqueeze_2093);  sub_631 = unsqueeze_2093 = None
    mul_2377: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_632, unsqueeze_2099);  sub_632 = unsqueeze_2099 = None
    mul_2378: "f32[432]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_280);  sum_217 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_200 = torch.ops.aten.convolution_backward.default(mul_2377, convolution_171, primals_368, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2377 = convolution_171 = primals_368 = None
    getitem_1086: "f32[8, 432, 21, 21]" = convolution_backward_200[0]
    getitem_1087: "f32[432, 432, 1, 1]" = convolution_backward_200[1];  convolution_backward_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_201 = torch.ops.aten.convolution_backward.default(getitem_1086, constant_pad_nd_24, primals_17, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1086 = constant_pad_nd_24 = primals_17 = None
    getitem_1089: "f32[8, 432, 45, 45]" = convolution_backward_201[0]
    getitem_1090: "f32[432, 1, 5, 5]" = convolution_backward_201[1];  convolution_backward_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_55: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(getitem_1089, [-1, -2, -1, -2]);  getitem_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_108: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_100, full_default, constant_pad_nd_55);  constant_pad_nd_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1142: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(add_1141, where_108);  add_1141 = where_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_22: "f32[8, 432, 43, 43]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_41, constant_pad_nd_23, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_225);  constant_pad_nd_23 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_56: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_22, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1143: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(add_1142, constant_pad_nd_56);  add_1142 = constant_pad_nd_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_218: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_41, [0, 2, 3])
    sub_633: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_170, unsqueeze_2102);  convolution_170 = unsqueeze_2102 = None
    mul_2379: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_41, sub_633)
    sum_219: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2379, [0, 2, 3]);  mul_2379 = None
    mul_2380: "f32[432]" = torch.ops.aten.mul.Tensor(sum_218, 0.0002834467120181406)
    unsqueeze_2103: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2380, 0);  mul_2380 = None
    unsqueeze_2104: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2103, 2);  unsqueeze_2103 = None
    unsqueeze_2105: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2104, 3);  unsqueeze_2104 = None
    mul_2381: "f32[432]" = torch.ops.aten.mul.Tensor(sum_219, 0.0002834467120181406)
    mul_2382: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_2383: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2381, mul_2382);  mul_2381 = mul_2382 = None
    unsqueeze_2106: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2383, 0);  mul_2383 = None
    unsqueeze_2107: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 2);  unsqueeze_2106 = None
    unsqueeze_2108: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2107, 3);  unsqueeze_2107 = None
    mul_2384: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_366);  primals_366 = None
    unsqueeze_2109: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2384, 0);  mul_2384 = None
    unsqueeze_2110: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2109, 2);  unsqueeze_2109 = None
    unsqueeze_2111: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2110, 3);  unsqueeze_2110 = None
    mul_2385: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_633, unsqueeze_2108);  sub_633 = unsqueeze_2108 = None
    sub_635: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_41, mul_2385);  slice_41 = mul_2385 = None
    sub_636: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_635, unsqueeze_2105);  sub_635 = unsqueeze_2105 = None
    mul_2386: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_636, unsqueeze_2111);  sub_636 = unsqueeze_2111 = None
    mul_2387: "f32[432]" = torch.ops.aten.mul.Tensor(sum_219, squeeze_277);  sum_219 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_202 = torch.ops.aten.convolution_backward.default(mul_2386, convolution_169, primals_365, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2386 = convolution_169 = primals_365 = None
    getitem_1092: "f32[8, 432, 21, 21]" = convolution_backward_202[0]
    getitem_1093: "f32[432, 432, 1, 1]" = convolution_backward_202[1];  convolution_backward_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_203 = torch.ops.aten.convolution_backward.default(getitem_1092, relu_90, primals_364, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1092 = primals_364 = None
    getitem_1095: "f32[8, 432, 21, 21]" = convolution_backward_203[0]
    getitem_1096: "f32[432, 1, 7, 7]" = convolution_backward_203[1];  convolution_backward_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_109: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
    where_109: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_109, full_default, getitem_1095);  le_109 = getitem_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_220: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_637: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_168, unsqueeze_2114);  convolution_168 = unsqueeze_2114 = None
    mul_2388: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_109, sub_637)
    sum_221: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2388, [0, 2, 3]);  mul_2388 = None
    mul_2389: "f32[432]" = torch.ops.aten.mul.Tensor(sum_220, 0.0002834467120181406)
    unsqueeze_2115: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2389, 0);  mul_2389 = None
    unsqueeze_2116: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2115, 2);  unsqueeze_2115 = None
    unsqueeze_2117: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2116, 3);  unsqueeze_2116 = None
    mul_2390: "f32[432]" = torch.ops.aten.mul.Tensor(sum_221, 0.0002834467120181406)
    mul_2391: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_2392: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2390, mul_2391);  mul_2390 = mul_2391 = None
    unsqueeze_2118: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2392, 0);  mul_2392 = None
    unsqueeze_2119: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 2);  unsqueeze_2118 = None
    unsqueeze_2120: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2119, 3);  unsqueeze_2119 = None
    mul_2393: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_362);  primals_362 = None
    unsqueeze_2121: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2393, 0);  mul_2393 = None
    unsqueeze_2122: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2121, 2);  unsqueeze_2121 = None
    unsqueeze_2123: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2122, 3);  unsqueeze_2122 = None
    mul_2394: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_637, unsqueeze_2120);  sub_637 = unsqueeze_2120 = None
    sub_639: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_109, mul_2394);  where_109 = mul_2394 = None
    sub_640: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_639, unsqueeze_2117);  sub_639 = unsqueeze_2117 = None
    mul_2395: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_640, unsqueeze_2123);  sub_640 = unsqueeze_2123 = None
    mul_2396: "f32[432]" = torch.ops.aten.mul.Tensor(sum_221, squeeze_274);  sum_221 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_204 = torch.ops.aten.convolution_backward.default(mul_2395, convolution_167, primals_361, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2395 = convolution_167 = primals_361 = None
    getitem_1098: "f32[8, 432, 21, 21]" = convolution_backward_204[0]
    getitem_1099: "f32[432, 432, 1, 1]" = convolution_backward_204[1];  convolution_backward_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_205 = torch.ops.aten.convolution_backward.default(getitem_1098, constant_pad_nd_22, primals_16, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1098 = constant_pad_nd_22 = primals_16 = None
    getitem_1101: "f32[8, 432, 47, 47]" = convolution_backward_205[0]
    getitem_1102: "f32[432, 1, 7, 7]" = convolution_backward_205[1];  convolution_backward_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_57: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(getitem_1101, [-2, -3, -2, -3]);  getitem_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_110: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_100, full_default, constant_pad_nd_57);  le_100 = constant_pad_nd_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1144: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(add_1143, where_110);  add_1143 = where_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_23: "f32[8, 432, 43, 43]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_40, constant_pad_nd_21, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_219);  constant_pad_nd_21 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_58: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_23, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1145: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(where_102, constant_pad_nd_58);  where_102 = constant_pad_nd_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_222: "f32[432]" = torch.ops.aten.sum.dim_IntList(slice_40, [0, 2, 3])
    sub_641: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_166, unsqueeze_2126);  convolution_166 = unsqueeze_2126 = None
    mul_2397: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(slice_40, sub_641)
    sum_223: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2397, [0, 2, 3]);  mul_2397 = None
    mul_2398: "f32[432]" = torch.ops.aten.mul.Tensor(sum_222, 0.0002834467120181406)
    unsqueeze_2127: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2398, 0);  mul_2398 = None
    unsqueeze_2128: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2127, 2);  unsqueeze_2127 = None
    unsqueeze_2129: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2128, 3);  unsqueeze_2128 = None
    mul_2399: "f32[432]" = torch.ops.aten.mul.Tensor(sum_223, 0.0002834467120181406)
    mul_2400: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_2401: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2399, mul_2400);  mul_2399 = mul_2400 = None
    unsqueeze_2130: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2401, 0);  mul_2401 = None
    unsqueeze_2131: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 2);  unsqueeze_2130 = None
    unsqueeze_2132: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2131, 3);  unsqueeze_2131 = None
    mul_2402: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_359);  primals_359 = None
    unsqueeze_2133: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2402, 0);  mul_2402 = None
    unsqueeze_2134: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2133, 2);  unsqueeze_2133 = None
    unsqueeze_2135: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2134, 3);  unsqueeze_2134 = None
    mul_2403: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_641, unsqueeze_2132);  sub_641 = unsqueeze_2132 = None
    sub_643: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(slice_40, mul_2403);  slice_40 = mul_2403 = None
    sub_644: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_643, unsqueeze_2129);  sub_643 = unsqueeze_2129 = None
    mul_2404: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_644, unsqueeze_2135);  sub_644 = unsqueeze_2135 = None
    mul_2405: "f32[432]" = torch.ops.aten.mul.Tensor(sum_223, squeeze_271);  sum_223 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_206 = torch.ops.aten.convolution_backward.default(mul_2404, convolution_165, primals_358, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2404 = convolution_165 = primals_358 = None
    getitem_1104: "f32[8, 432, 21, 21]" = convolution_backward_206[0]
    getitem_1105: "f32[432, 432, 1, 1]" = convolution_backward_206[1];  convolution_backward_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_207 = torch.ops.aten.convolution_backward.default(getitem_1104, relu_88, primals_357, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1104 = primals_357 = None
    getitem_1107: "f32[8, 432, 21, 21]" = convolution_backward_207[0]
    getitem_1108: "f32[432, 1, 5, 5]" = convolution_backward_207[1];  convolution_backward_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_111: "b8[8, 432, 21, 21]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
    where_111: "f32[8, 432, 21, 21]" = torch.ops.aten.where.self(le_111, full_default, getitem_1107);  le_111 = getitem_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_224: "f32[432]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_645: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(convolution_164, unsqueeze_2138);  convolution_164 = unsqueeze_2138 = None
    mul_2406: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(where_111, sub_645)
    sum_225: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2406, [0, 2, 3]);  mul_2406 = None
    mul_2407: "f32[432]" = torch.ops.aten.mul.Tensor(sum_224, 0.0002834467120181406)
    unsqueeze_2139: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2407, 0);  mul_2407 = None
    unsqueeze_2140: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2139, 2);  unsqueeze_2139 = None
    unsqueeze_2141: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2140, 3);  unsqueeze_2140 = None
    mul_2408: "f32[432]" = torch.ops.aten.mul.Tensor(sum_225, 0.0002834467120181406)
    mul_2409: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_2410: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2408, mul_2409);  mul_2408 = mul_2409 = None
    unsqueeze_2142: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2410, 0);  mul_2410 = None
    unsqueeze_2143: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 2);  unsqueeze_2142 = None
    unsqueeze_2144: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2143, 3);  unsqueeze_2143 = None
    mul_2411: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_355);  primals_355 = None
    unsqueeze_2145: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2411, 0);  mul_2411 = None
    unsqueeze_2146: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2145, 2);  unsqueeze_2145 = None
    unsqueeze_2147: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2146, 3);  unsqueeze_2146 = None
    mul_2412: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_645, unsqueeze_2144);  sub_645 = unsqueeze_2144 = None
    sub_647: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(where_111, mul_2412);  where_111 = mul_2412 = None
    sub_648: "f32[8, 432, 21, 21]" = torch.ops.aten.sub.Tensor(sub_647, unsqueeze_2141);  sub_647 = unsqueeze_2141 = None
    mul_2413: "f32[8, 432, 21, 21]" = torch.ops.aten.mul.Tensor(sub_648, unsqueeze_2147);  sub_648 = unsqueeze_2147 = None
    mul_2414: "f32[432]" = torch.ops.aten.mul.Tensor(sum_225, squeeze_268);  sum_225 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_208 = torch.ops.aten.convolution_backward.default(mul_2413, convolution_163, primals_354, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2413 = convolution_163 = primals_354 = None
    getitem_1110: "f32[8, 432, 21, 21]" = convolution_backward_208[0]
    getitem_1111: "f32[432, 432, 1, 1]" = convolution_backward_208[1];  convolution_backward_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_209 = torch.ops.aten.convolution_backward.default(getitem_1110, constant_pad_nd_20, primals_15, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 432, [True, True, False]);  getitem_1110 = constant_pad_nd_20 = primals_15 = None
    getitem_1113: "f32[8, 432, 45, 45]" = convolution_backward_209[0]
    getitem_1114: "f32[432, 1, 5, 5]" = convolution_backward_209[1];  convolution_backward_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_59: "f32[8, 432, 42, 42]" = torch.ops.aten.constant_pad_nd.default(getitem_1113, [-1, -2, -1, -2]);  getitem_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_112: "f32[8, 432, 42, 42]" = torch.ops.aten.where.self(le_102, full_default, constant_pad_nd_59);  le_102 = constant_pad_nd_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1146: "f32[8, 432, 42, 42]" = torch.ops.aten.add.Tensor(add_1145, where_112);  add_1145 = where_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_226: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1144, [0, 2, 3])
    sub_649: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_162, unsqueeze_2150);  convolution_162 = unsqueeze_2150 = None
    mul_2415: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(add_1144, sub_649)
    sum_227: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2415, [0, 2, 3]);  mul_2415 = None
    mul_2416: "f32[432]" = torch.ops.aten.mul.Tensor(sum_226, 7.086167800453515e-05)
    unsqueeze_2151: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2416, 0);  mul_2416 = None
    unsqueeze_2152: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2151, 2);  unsqueeze_2151 = None
    unsqueeze_2153: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2152, 3);  unsqueeze_2152 = None
    mul_2417: "f32[432]" = torch.ops.aten.mul.Tensor(sum_227, 7.086167800453515e-05)
    mul_2418: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_2419: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2417, mul_2418);  mul_2417 = mul_2418 = None
    unsqueeze_2154: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2419, 0);  mul_2419 = None
    unsqueeze_2155: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, 2);  unsqueeze_2154 = None
    unsqueeze_2156: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2155, 3);  unsqueeze_2155 = None
    mul_2420: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_352);  primals_352 = None
    unsqueeze_2157: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2420, 0);  mul_2420 = None
    unsqueeze_2158: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2157, 2);  unsqueeze_2157 = None
    unsqueeze_2159: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2158, 3);  unsqueeze_2158 = None
    mul_2421: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_649, unsqueeze_2156);  sub_649 = unsqueeze_2156 = None
    sub_651: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(add_1144, mul_2421);  add_1144 = mul_2421 = None
    sub_652: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(sub_651, unsqueeze_2153);  sub_651 = unsqueeze_2153 = None
    mul_2422: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_652, unsqueeze_2159);  sub_652 = unsqueeze_2159 = None
    mul_2423: "f32[432]" = torch.ops.aten.mul.Tensor(sum_227, squeeze_265);  sum_227 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_210 = torch.ops.aten.convolution_backward.default(mul_2422, relu_86, primals_351, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2422 = relu_86 = primals_351 = None
    getitem_1116: "f32[8, 1080, 42, 42]" = convolution_backward_210[0]
    getitem_1117: "f32[432, 1080, 1, 1]" = convolution_backward_210[1];  convolution_backward_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_113: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_99, full_default, getitem_1116);  le_99 = getitem_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1147: "f32[8, 1080, 42, 42]" = torch.ops.aten.add.Tensor(where_99, where_113);  where_99 = where_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_228: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_1146, [0, 2, 3])
    sub_653: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_161, unsqueeze_2162);  convolution_161 = unsqueeze_2162 = None
    mul_2424: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(add_1146, sub_653)
    sum_229: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_2424, [0, 2, 3]);  mul_2424 = None
    mul_2425: "f32[432]" = torch.ops.aten.mul.Tensor(sum_228, 7.086167800453515e-05)
    unsqueeze_2163: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2425, 0);  mul_2425 = None
    unsqueeze_2164: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2163, 2);  unsqueeze_2163 = None
    unsqueeze_2165: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2164, 3);  unsqueeze_2164 = None
    mul_2426: "f32[432]" = torch.ops.aten.mul.Tensor(sum_229, 7.086167800453515e-05)
    mul_2427: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_2428: "f32[432]" = torch.ops.aten.mul.Tensor(mul_2426, mul_2427);  mul_2426 = mul_2427 = None
    unsqueeze_2166: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2428, 0);  mul_2428 = None
    unsqueeze_2167: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, 2);  unsqueeze_2166 = None
    unsqueeze_2168: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2167, 3);  unsqueeze_2167 = None
    mul_2429: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_349);  primals_349 = None
    unsqueeze_2169: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_2429, 0);  mul_2429 = None
    unsqueeze_2170: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2169, 2);  unsqueeze_2169 = None
    unsqueeze_2171: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2170, 3);  unsqueeze_2170 = None
    mul_2430: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_653, unsqueeze_2168);  sub_653 = unsqueeze_2168 = None
    sub_655: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(add_1146, mul_2430);  add_1146 = mul_2430 = None
    sub_656: "f32[8, 432, 42, 42]" = torch.ops.aten.sub.Tensor(sub_655, unsqueeze_2165);  sub_655 = unsqueeze_2165 = None
    mul_2431: "f32[8, 432, 42, 42]" = torch.ops.aten.mul.Tensor(sub_656, unsqueeze_2171);  sub_656 = unsqueeze_2171 = None
    mul_2432: "f32[432]" = torch.ops.aten.mul.Tensor(sum_229, squeeze_262);  sum_229 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_211 = torch.ops.aten.convolution_backward.default(mul_2431, relu_72, primals_348, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2431 = primals_348 = None
    getitem_1119: "f32[8, 1080, 42, 42]" = convolution_backward_211[0]
    getitem_1120: "f32[432, 1080, 1, 1]" = convolution_backward_211[1];  convolution_backward_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_114: "b8[8, 1080, 42, 42]" = torch.ops.aten.le.Scalar(relu_72, 0)
    where_114: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_114, full_default, getitem_1119);  getitem_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_45: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1147, 1, 0, 216)
    slice_46: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1147, 1, 216, 432)
    slice_47: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1147, 1, 432, 648)
    slice_48: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1147, 1, 648, 864)
    slice_49: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1147, 1, 864, 1080);  add_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_230: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_49, [0, 2, 3])
    sub_657: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_160, unsqueeze_2174);  convolution_160 = unsqueeze_2174 = None
    mul_2433: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_49, sub_657)
    sum_231: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2433, [0, 2, 3]);  mul_2433 = None
    mul_2434: "f32[216]" = torch.ops.aten.mul.Tensor(sum_230, 7.086167800453515e-05)
    unsqueeze_2175: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2434, 0);  mul_2434 = None
    unsqueeze_2176: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2175, 2);  unsqueeze_2175 = None
    unsqueeze_2177: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2176, 3);  unsqueeze_2176 = None
    mul_2435: "f32[216]" = torch.ops.aten.mul.Tensor(sum_231, 7.086167800453515e-05)
    mul_2436: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_2437: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2435, mul_2436);  mul_2435 = mul_2436 = None
    unsqueeze_2178: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2437, 0);  mul_2437 = None
    unsqueeze_2179: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 2);  unsqueeze_2178 = None
    unsqueeze_2180: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2179, 3);  unsqueeze_2179 = None
    mul_2438: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_346);  primals_346 = None
    unsqueeze_2181: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2438, 0);  mul_2438 = None
    unsqueeze_2182: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2181, 2);  unsqueeze_2181 = None
    unsqueeze_2183: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2182, 3);  unsqueeze_2182 = None
    mul_2439: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_657, unsqueeze_2180);  sub_657 = unsqueeze_2180 = None
    sub_659: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_49, mul_2439);  mul_2439 = None
    sub_660: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_659, unsqueeze_2177);  sub_659 = unsqueeze_2177 = None
    mul_2440: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_660, unsqueeze_2183);  sub_660 = unsqueeze_2183 = None
    mul_2441: "f32[216]" = torch.ops.aten.mul.Tensor(sum_231, squeeze_259);  sum_231 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_212 = torch.ops.aten.convolution_backward.default(mul_2440, convolution_159, primals_345, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2440 = convolution_159 = primals_345 = None
    getitem_1122: "f32[8, 216, 42, 42]" = convolution_backward_212[0]
    getitem_1123: "f32[216, 216, 1, 1]" = convolution_backward_212[1];  convolution_backward_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_213 = torch.ops.aten.convolution_backward.default(getitem_1122, relu_84, primals_344, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1122 = primals_344 = None
    getitem_1125: "f32[8, 216, 42, 42]" = convolution_backward_213[0]
    getitem_1126: "f32[216, 1, 3, 3]" = convolution_backward_213[1];  convolution_backward_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_115: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
    where_115: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_115, full_default, getitem_1125);  le_115 = getitem_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_232: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_661: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_158, unsqueeze_2186);  convolution_158 = unsqueeze_2186 = None
    mul_2442: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_115, sub_661)
    sum_233: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2442, [0, 2, 3]);  mul_2442 = None
    mul_2443: "f32[216]" = torch.ops.aten.mul.Tensor(sum_232, 7.086167800453515e-05)
    unsqueeze_2187: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2443, 0);  mul_2443 = None
    unsqueeze_2188: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2187, 2);  unsqueeze_2187 = None
    unsqueeze_2189: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2188, 3);  unsqueeze_2188 = None
    mul_2444: "f32[216]" = torch.ops.aten.mul.Tensor(sum_233, 7.086167800453515e-05)
    mul_2445: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_2446: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2444, mul_2445);  mul_2444 = mul_2445 = None
    unsqueeze_2190: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2446, 0);  mul_2446 = None
    unsqueeze_2191: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 2);  unsqueeze_2190 = None
    unsqueeze_2192: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2191, 3);  unsqueeze_2191 = None
    mul_2447: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_342);  primals_342 = None
    unsqueeze_2193: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2447, 0);  mul_2447 = None
    unsqueeze_2194: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2193, 2);  unsqueeze_2193 = None
    unsqueeze_2195: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2194, 3);  unsqueeze_2194 = None
    mul_2448: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_661, unsqueeze_2192);  sub_661 = unsqueeze_2192 = None
    sub_663: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_115, mul_2448);  where_115 = mul_2448 = None
    sub_664: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_663, unsqueeze_2189);  sub_663 = unsqueeze_2189 = None
    mul_2449: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_664, unsqueeze_2195);  sub_664 = unsqueeze_2195 = None
    mul_2450: "f32[216]" = torch.ops.aten.mul.Tensor(sum_233, squeeze_256);  sum_233 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_214 = torch.ops.aten.convolution_backward.default(mul_2449, convolution_157, primals_341, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2449 = convolution_157 = primals_341 = None
    getitem_1128: "f32[8, 216, 42, 42]" = convolution_backward_214[0]
    getitem_1129: "f32[216, 216, 1, 1]" = convolution_backward_214[1];  convolution_backward_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_215 = torch.ops.aten.convolution_backward.default(getitem_1128, relu_73, primals_340, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1128 = primals_340 = None
    getitem_1131: "f32[8, 216, 42, 42]" = convolution_backward_215[0]
    getitem_1132: "f32[216, 1, 3, 3]" = convolution_backward_215[1];  convolution_backward_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_116: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_73, 0)
    where_116: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_116, full_default, getitem_1131);  getitem_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_24: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_48, add_399, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1148: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_49, max_pool2d_with_indices_backward_24);  slice_49 = max_pool2d_with_indices_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_234: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_48, [0, 2, 3])
    sub_665: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_156, unsqueeze_2198);  convolution_156 = unsqueeze_2198 = None
    mul_2451: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_48, sub_665)
    sum_235: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2451, [0, 2, 3]);  mul_2451 = None
    mul_2452: "f32[216]" = torch.ops.aten.mul.Tensor(sum_234, 7.086167800453515e-05)
    unsqueeze_2199: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2452, 0);  mul_2452 = None
    unsqueeze_2200: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2199, 2);  unsqueeze_2199 = None
    unsqueeze_2201: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2200, 3);  unsqueeze_2200 = None
    mul_2453: "f32[216]" = torch.ops.aten.mul.Tensor(sum_235, 7.086167800453515e-05)
    mul_2454: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_2455: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2453, mul_2454);  mul_2453 = mul_2454 = None
    unsqueeze_2202: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2455, 0);  mul_2455 = None
    unsqueeze_2203: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 2);  unsqueeze_2202 = None
    unsqueeze_2204: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2203, 3);  unsqueeze_2203 = None
    mul_2456: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_338);  primals_338 = None
    unsqueeze_2205: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2456, 0);  mul_2456 = None
    unsqueeze_2206: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2205, 2);  unsqueeze_2205 = None
    unsqueeze_2207: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2206, 3);  unsqueeze_2206 = None
    mul_2457: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_665, unsqueeze_2204);  sub_665 = unsqueeze_2204 = None
    sub_667: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_48, mul_2457);  slice_48 = mul_2457 = None
    sub_668: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_667, unsqueeze_2201);  sub_667 = unsqueeze_2201 = None
    mul_2458: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_668, unsqueeze_2207);  sub_668 = unsqueeze_2207 = None
    mul_2459: "f32[216]" = torch.ops.aten.mul.Tensor(sum_235, squeeze_253);  sum_235 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_216 = torch.ops.aten.convolution_backward.default(mul_2458, convolution_155, primals_337, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2458 = convolution_155 = primals_337 = None
    getitem_1134: "f32[8, 216, 42, 42]" = convolution_backward_216[0]
    getitem_1135: "f32[216, 216, 1, 1]" = convolution_backward_216[1];  convolution_backward_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_217 = torch.ops.aten.convolution_backward.default(getitem_1134, relu_82, primals_336, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1134 = primals_336 = None
    getitem_1137: "f32[8, 216, 42, 42]" = convolution_backward_217[0]
    getitem_1138: "f32[216, 1, 3, 3]" = convolution_backward_217[1];  convolution_backward_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_117: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
    where_117: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_117, full_default, getitem_1137);  le_117 = getitem_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_236: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_669: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_2210);  convolution_154 = unsqueeze_2210 = None
    mul_2460: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_117, sub_669)
    sum_237: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2460, [0, 2, 3]);  mul_2460 = None
    mul_2461: "f32[216]" = torch.ops.aten.mul.Tensor(sum_236, 7.086167800453515e-05)
    unsqueeze_2211: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2461, 0);  mul_2461 = None
    unsqueeze_2212: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2211, 2);  unsqueeze_2211 = None
    unsqueeze_2213: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2212, 3);  unsqueeze_2212 = None
    mul_2462: "f32[216]" = torch.ops.aten.mul.Tensor(sum_237, 7.086167800453515e-05)
    mul_2463: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_2464: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2462, mul_2463);  mul_2462 = mul_2463 = None
    unsqueeze_2214: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2464, 0);  mul_2464 = None
    unsqueeze_2215: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 2);  unsqueeze_2214 = None
    unsqueeze_2216: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2215, 3);  unsqueeze_2215 = None
    mul_2465: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_334);  primals_334 = None
    unsqueeze_2217: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2465, 0);  mul_2465 = None
    unsqueeze_2218: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2217, 2);  unsqueeze_2217 = None
    unsqueeze_2219: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2218, 3);  unsqueeze_2218 = None
    mul_2466: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_669, unsqueeze_2216);  sub_669 = unsqueeze_2216 = None
    sub_671: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_117, mul_2466);  where_117 = mul_2466 = None
    sub_672: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_671, unsqueeze_2213);  sub_671 = unsqueeze_2213 = None
    mul_2467: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_672, unsqueeze_2219);  sub_672 = unsqueeze_2219 = None
    mul_2468: "f32[216]" = torch.ops.aten.mul.Tensor(sum_237, squeeze_250);  sum_237 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_218 = torch.ops.aten.convolution_backward.default(mul_2467, convolution_153, primals_333, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2467 = convolution_153 = primals_333 = None
    getitem_1140: "f32[8, 216, 42, 42]" = convolution_backward_218[0]
    getitem_1141: "f32[216, 216, 1, 1]" = convolution_backward_218[1];  convolution_backward_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_219 = torch.ops.aten.convolution_backward.default(getitem_1140, relu_81, primals_332, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1140 = primals_332 = None
    getitem_1143: "f32[8, 216, 42, 42]" = convolution_backward_219[0]
    getitem_1144: "f32[216, 1, 3, 3]" = convolution_backward_219[1];  convolution_backward_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_118: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
    where_118: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_118, full_default, getitem_1143);  le_118 = getitem_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1149: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_47, where_118);  slice_47 = where_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_238: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1149, [0, 2, 3])
    sub_673: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_2222);  convolution_152 = unsqueeze_2222 = None
    mul_2469: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1149, sub_673)
    sum_239: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2469, [0, 2, 3]);  mul_2469 = None
    mul_2470: "f32[216]" = torch.ops.aten.mul.Tensor(sum_238, 7.086167800453515e-05)
    unsqueeze_2223: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2470, 0);  mul_2470 = None
    unsqueeze_2224: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2223, 2);  unsqueeze_2223 = None
    unsqueeze_2225: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2224, 3);  unsqueeze_2224 = None
    mul_2471: "f32[216]" = torch.ops.aten.mul.Tensor(sum_239, 7.086167800453515e-05)
    mul_2472: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_2473: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2471, mul_2472);  mul_2471 = mul_2472 = None
    unsqueeze_2226: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2473, 0);  mul_2473 = None
    unsqueeze_2227: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 2);  unsqueeze_2226 = None
    unsqueeze_2228: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2227, 3);  unsqueeze_2227 = None
    mul_2474: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_330);  primals_330 = None
    unsqueeze_2229: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2474, 0);  mul_2474 = None
    unsqueeze_2230: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2229, 2);  unsqueeze_2229 = None
    unsqueeze_2231: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2230, 3);  unsqueeze_2230 = None
    mul_2475: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_673, unsqueeze_2228);  sub_673 = unsqueeze_2228 = None
    sub_675: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1149, mul_2475);  mul_2475 = None
    sub_676: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_675, unsqueeze_2225);  sub_675 = None
    mul_2476: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_676, unsqueeze_2231);  sub_676 = unsqueeze_2231 = None
    mul_2477: "f32[216]" = torch.ops.aten.mul.Tensor(sum_239, squeeze_247);  sum_239 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_220 = torch.ops.aten.convolution_backward.default(mul_2476, convolution_151, primals_329, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2476 = convolution_151 = primals_329 = None
    getitem_1146: "f32[8, 216, 42, 42]" = convolution_backward_220[0]
    getitem_1147: "f32[216, 216, 1, 1]" = convolution_backward_220[1];  convolution_backward_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_221 = torch.ops.aten.convolution_backward.default(getitem_1146, relu_80, primals_328, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1146 = primals_328 = None
    getitem_1149: "f32[8, 216, 42, 42]" = convolution_backward_221[0]
    getitem_1150: "f32[216, 1, 3, 3]" = convolution_backward_221[1];  convolution_backward_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_119: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
    where_119: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_119, full_default, getitem_1149);  le_119 = getitem_1149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_240: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_677: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_2234);  convolution_150 = unsqueeze_2234 = None
    mul_2478: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_119, sub_677)
    sum_241: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2478, [0, 2, 3]);  mul_2478 = None
    mul_2479: "f32[216]" = torch.ops.aten.mul.Tensor(sum_240, 7.086167800453515e-05)
    unsqueeze_2235: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2479, 0);  mul_2479 = None
    unsqueeze_2236: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2235, 2);  unsqueeze_2235 = None
    unsqueeze_2237: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2236, 3);  unsqueeze_2236 = None
    mul_2480: "f32[216]" = torch.ops.aten.mul.Tensor(sum_241, 7.086167800453515e-05)
    mul_2481: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_2482: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2480, mul_2481);  mul_2480 = mul_2481 = None
    unsqueeze_2238: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2482, 0);  mul_2482 = None
    unsqueeze_2239: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, 2);  unsqueeze_2238 = None
    unsqueeze_2240: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2239, 3);  unsqueeze_2239 = None
    mul_2483: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_326);  primals_326 = None
    unsqueeze_2241: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2483, 0);  mul_2483 = None
    unsqueeze_2242: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2241, 2);  unsqueeze_2241 = None
    unsqueeze_2243: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2242, 3);  unsqueeze_2242 = None
    mul_2484: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_677, unsqueeze_2240);  sub_677 = unsqueeze_2240 = None
    sub_679: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_119, mul_2484);  where_119 = mul_2484 = None
    sub_680: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_679, unsqueeze_2237);  sub_679 = unsqueeze_2237 = None
    mul_2485: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_680, unsqueeze_2243);  sub_680 = unsqueeze_2243 = None
    mul_2486: "f32[216]" = torch.ops.aten.mul.Tensor(sum_241, squeeze_244);  sum_241 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_222 = torch.ops.aten.convolution_backward.default(mul_2485, convolution_149, primals_325, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2485 = convolution_149 = primals_325 = None
    getitem_1152: "f32[8, 216, 42, 42]" = convolution_backward_222[0]
    getitem_1153: "f32[216, 216, 1, 1]" = convolution_backward_222[1];  convolution_backward_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_223 = torch.ops.aten.convolution_backward.default(getitem_1152, relu_75, primals_324, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1152 = primals_324 = None
    getitem_1155: "f32[8, 216, 42, 42]" = convolution_backward_223[0]
    getitem_1156: "f32[216, 1, 3, 3]" = convolution_backward_223[1];  convolution_backward_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_120: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_75, 0)
    where_120: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_120, full_default, getitem_1155);  getitem_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1150: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1148, where_120);  add_1148 = where_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_681: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_2246);  convolution_148 = unsqueeze_2246 = None
    mul_2487: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1149, sub_681)
    sum_243: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2487, [0, 2, 3]);  mul_2487 = None
    mul_2489: "f32[216]" = torch.ops.aten.mul.Tensor(sum_243, 7.086167800453515e-05)
    mul_2490: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_2491: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2489, mul_2490);  mul_2489 = mul_2490 = None
    unsqueeze_2250: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2491, 0);  mul_2491 = None
    unsqueeze_2251: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, 2);  unsqueeze_2250 = None
    unsqueeze_2252: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2251, 3);  unsqueeze_2251 = None
    mul_2492: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_322);  primals_322 = None
    unsqueeze_2253: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2492, 0);  mul_2492 = None
    unsqueeze_2254: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2253, 2);  unsqueeze_2253 = None
    unsqueeze_2255: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2254, 3);  unsqueeze_2254 = None
    mul_2493: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_681, unsqueeze_2252);  sub_681 = unsqueeze_2252 = None
    sub_683: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1149, mul_2493);  add_1149 = mul_2493 = None
    sub_684: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_683, unsqueeze_2225);  sub_683 = unsqueeze_2225 = None
    mul_2494: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_684, unsqueeze_2255);  sub_684 = unsqueeze_2255 = None
    mul_2495: "f32[216]" = torch.ops.aten.mul.Tensor(sum_243, squeeze_241);  sum_243 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_224 = torch.ops.aten.convolution_backward.default(mul_2494, convolution_147, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2494 = convolution_147 = primals_321 = None
    getitem_1158: "f32[8, 216, 42, 42]" = convolution_backward_224[0]
    getitem_1159: "f32[216, 216, 1, 1]" = convolution_backward_224[1];  convolution_backward_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_225 = torch.ops.aten.convolution_backward.default(getitem_1158, relu_78, primals_320, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1158 = primals_320 = None
    getitem_1161: "f32[8, 216, 42, 42]" = convolution_backward_225[0]
    getitem_1162: "f32[216, 1, 5, 5]" = convolution_backward_225[1];  convolution_backward_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_121: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
    where_121: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_121, full_default, getitem_1161);  le_121 = getitem_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_244: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_685: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_2258);  convolution_146 = unsqueeze_2258 = None
    mul_2496: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_121, sub_685)
    sum_245: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2496, [0, 2, 3]);  mul_2496 = None
    mul_2497: "f32[216]" = torch.ops.aten.mul.Tensor(sum_244, 7.086167800453515e-05)
    unsqueeze_2259: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2497, 0);  mul_2497 = None
    unsqueeze_2260: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2259, 2);  unsqueeze_2259 = None
    unsqueeze_2261: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2260, 3);  unsqueeze_2260 = None
    mul_2498: "f32[216]" = torch.ops.aten.mul.Tensor(sum_245, 7.086167800453515e-05)
    mul_2499: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_2500: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2498, mul_2499);  mul_2498 = mul_2499 = None
    unsqueeze_2262: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2500, 0);  mul_2500 = None
    unsqueeze_2263: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, 2);  unsqueeze_2262 = None
    unsqueeze_2264: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2263, 3);  unsqueeze_2263 = None
    mul_2501: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_318);  primals_318 = None
    unsqueeze_2265: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2501, 0);  mul_2501 = None
    unsqueeze_2266: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2265, 2);  unsqueeze_2265 = None
    unsqueeze_2267: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2266, 3);  unsqueeze_2266 = None
    mul_2502: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_685, unsqueeze_2264);  sub_685 = unsqueeze_2264 = None
    sub_687: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_121, mul_2502);  where_121 = mul_2502 = None
    sub_688: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_687, unsqueeze_2261);  sub_687 = unsqueeze_2261 = None
    mul_2503: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_688, unsqueeze_2267);  sub_688 = unsqueeze_2267 = None
    mul_2504: "f32[216]" = torch.ops.aten.mul.Tensor(sum_245, squeeze_238);  sum_245 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_226 = torch.ops.aten.convolution_backward.default(mul_2503, convolution_145, primals_317, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2503 = convolution_145 = primals_317 = None
    getitem_1164: "f32[8, 216, 42, 42]" = convolution_backward_226[0]
    getitem_1165: "f32[216, 216, 1, 1]" = convolution_backward_226[1];  convolution_backward_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_227 = torch.ops.aten.convolution_backward.default(getitem_1164, relu_75, primals_316, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1164 = primals_316 = None
    getitem_1167: "f32[8, 216, 42, 42]" = convolution_backward_227[0]
    getitem_1168: "f32[216, 1, 5, 5]" = convolution_backward_227[1];  convolution_backward_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_122: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_120, full_default, getitem_1167);  getitem_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1151: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1150, where_122);  add_1150 = where_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_25: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_46, add_399, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_191);  add_399 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1152: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1151, max_pool2d_with_indices_backward_25);  add_1151 = max_pool2d_with_indices_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_246: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_46, [0, 2, 3])
    sub_689: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_2270);  convolution_144 = unsqueeze_2270 = None
    mul_2505: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_46, sub_689)
    sum_247: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2505, [0, 2, 3]);  mul_2505 = None
    mul_2506: "f32[216]" = torch.ops.aten.mul.Tensor(sum_246, 7.086167800453515e-05)
    unsqueeze_2271: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2506, 0);  mul_2506 = None
    unsqueeze_2272: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2271, 2);  unsqueeze_2271 = None
    unsqueeze_2273: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2272, 3);  unsqueeze_2272 = None
    mul_2507: "f32[216]" = torch.ops.aten.mul.Tensor(sum_247, 7.086167800453515e-05)
    mul_2508: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_2509: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2507, mul_2508);  mul_2507 = mul_2508 = None
    unsqueeze_2274: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2509, 0);  mul_2509 = None
    unsqueeze_2275: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, 2);  unsqueeze_2274 = None
    unsqueeze_2276: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2275, 3);  unsqueeze_2275 = None
    mul_2510: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_314);  primals_314 = None
    unsqueeze_2277: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2510, 0);  mul_2510 = None
    unsqueeze_2278: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2277, 2);  unsqueeze_2277 = None
    unsqueeze_2279: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2278, 3);  unsqueeze_2278 = None
    mul_2511: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_689, unsqueeze_2276);  sub_689 = unsqueeze_2276 = None
    sub_691: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_46, mul_2511);  slice_46 = mul_2511 = None
    sub_692: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_691, unsqueeze_2273);  sub_691 = unsqueeze_2273 = None
    mul_2512: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_692, unsqueeze_2279);  sub_692 = unsqueeze_2279 = None
    mul_2513: "f32[216]" = torch.ops.aten.mul.Tensor(sum_247, squeeze_235);  sum_247 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_228 = torch.ops.aten.convolution_backward.default(mul_2512, convolution_143, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2512 = convolution_143 = primals_313 = None
    getitem_1170: "f32[8, 216, 42, 42]" = convolution_backward_228[0]
    getitem_1171: "f32[216, 216, 1, 1]" = convolution_backward_228[1];  convolution_backward_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_229 = torch.ops.aten.convolution_backward.default(getitem_1170, relu_76, primals_312, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1170 = primals_312 = None
    getitem_1173: "f32[8, 216, 42, 42]" = convolution_backward_229[0]
    getitem_1174: "f32[216, 1, 7, 7]" = convolution_backward_229[1];  convolution_backward_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_123: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    where_123: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_123, full_default, getitem_1173);  le_123 = getitem_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_248: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_693: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_2282);  convolution_142 = unsqueeze_2282 = None
    mul_2514: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_123, sub_693)
    sum_249: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2514, [0, 2, 3]);  mul_2514 = None
    mul_2515: "f32[216]" = torch.ops.aten.mul.Tensor(sum_248, 7.086167800453515e-05)
    unsqueeze_2283: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2515, 0);  mul_2515 = None
    unsqueeze_2284: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2283, 2);  unsqueeze_2283 = None
    unsqueeze_2285: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2284, 3);  unsqueeze_2284 = None
    mul_2516: "f32[216]" = torch.ops.aten.mul.Tensor(sum_249, 7.086167800453515e-05)
    mul_2517: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_2518: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2516, mul_2517);  mul_2516 = mul_2517 = None
    unsqueeze_2286: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2518, 0);  mul_2518 = None
    unsqueeze_2287: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, 2);  unsqueeze_2286 = None
    unsqueeze_2288: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2287, 3);  unsqueeze_2287 = None
    mul_2519: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_310);  primals_310 = None
    unsqueeze_2289: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2519, 0);  mul_2519 = None
    unsqueeze_2290: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2289, 2);  unsqueeze_2289 = None
    unsqueeze_2291: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2290, 3);  unsqueeze_2290 = None
    mul_2520: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_693, unsqueeze_2288);  sub_693 = unsqueeze_2288 = None
    sub_695: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_123, mul_2520);  where_123 = mul_2520 = None
    sub_696: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_695, unsqueeze_2285);  sub_695 = unsqueeze_2285 = None
    mul_2521: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_696, unsqueeze_2291);  sub_696 = unsqueeze_2291 = None
    mul_2522: "f32[216]" = torch.ops.aten.mul.Tensor(sum_249, squeeze_232);  sum_249 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_230 = torch.ops.aten.convolution_backward.default(mul_2521, convolution_141, primals_309, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2521 = convolution_141 = primals_309 = None
    getitem_1176: "f32[8, 216, 42, 42]" = convolution_backward_230[0]
    getitem_1177: "f32[216, 216, 1, 1]" = convolution_backward_230[1];  convolution_backward_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_231 = torch.ops.aten.convolution_backward.default(getitem_1176, relu_75, primals_308, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1176 = relu_75 = primals_308 = None
    getitem_1179: "f32[8, 216, 42, 42]" = convolution_backward_231[0]
    getitem_1180: "f32[216, 1, 7, 7]" = convolution_backward_231[1];  convolution_backward_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_124: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_120, full_default, getitem_1179);  le_120 = getitem_1179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1153: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1152, where_124);  add_1152 = where_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_26: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_45, add_394, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_185);  add_394 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1154: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(where_116, max_pool2d_with_indices_backward_26);  where_116 = max_pool2d_with_indices_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_250: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_45, [0, 2, 3])
    sub_697: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_2294);  convolution_140 = unsqueeze_2294 = None
    mul_2523: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_45, sub_697)
    sum_251: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2523, [0, 2, 3]);  mul_2523 = None
    mul_2524: "f32[216]" = torch.ops.aten.mul.Tensor(sum_250, 7.086167800453515e-05)
    unsqueeze_2295: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2524, 0);  mul_2524 = None
    unsqueeze_2296: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2295, 2);  unsqueeze_2295 = None
    unsqueeze_2297: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2296, 3);  unsqueeze_2296 = None
    mul_2525: "f32[216]" = torch.ops.aten.mul.Tensor(sum_251, 7.086167800453515e-05)
    mul_2526: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_2527: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2525, mul_2526);  mul_2525 = mul_2526 = None
    unsqueeze_2298: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2527, 0);  mul_2527 = None
    unsqueeze_2299: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, 2);  unsqueeze_2298 = None
    unsqueeze_2300: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2299, 3);  unsqueeze_2299 = None
    mul_2528: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_306);  primals_306 = None
    unsqueeze_2301: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2528, 0);  mul_2528 = None
    unsqueeze_2302: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2301, 2);  unsqueeze_2301 = None
    unsqueeze_2303: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2302, 3);  unsqueeze_2302 = None
    mul_2529: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_697, unsqueeze_2300);  sub_697 = unsqueeze_2300 = None
    sub_699: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_45, mul_2529);  slice_45 = mul_2529 = None
    sub_700: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_699, unsqueeze_2297);  sub_699 = unsqueeze_2297 = None
    mul_2530: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_700, unsqueeze_2303);  sub_700 = unsqueeze_2303 = None
    mul_2531: "f32[216]" = torch.ops.aten.mul.Tensor(sum_251, squeeze_229);  sum_251 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_232 = torch.ops.aten.convolution_backward.default(mul_2530, convolution_139, primals_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2530 = convolution_139 = primals_305 = None
    getitem_1182: "f32[8, 216, 42, 42]" = convolution_backward_232[0]
    getitem_1183: "f32[216, 216, 1, 1]" = convolution_backward_232[1];  convolution_backward_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_233 = torch.ops.aten.convolution_backward.default(getitem_1182, relu_74, primals_304, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1182 = primals_304 = None
    getitem_1185: "f32[8, 216, 42, 42]" = convolution_backward_233[0]
    getitem_1186: "f32[216, 1, 5, 5]" = convolution_backward_233[1];  convolution_backward_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_125: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
    where_125: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_125, full_default, getitem_1185);  le_125 = getitem_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_252: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_701: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_2306);  convolution_138 = unsqueeze_2306 = None
    mul_2532: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_125, sub_701)
    sum_253: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2532, [0, 2, 3]);  mul_2532 = None
    mul_2533: "f32[216]" = torch.ops.aten.mul.Tensor(sum_252, 7.086167800453515e-05)
    unsqueeze_2307: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2533, 0);  mul_2533 = None
    unsqueeze_2308: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2307, 2);  unsqueeze_2307 = None
    unsqueeze_2309: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2308, 3);  unsqueeze_2308 = None
    mul_2534: "f32[216]" = torch.ops.aten.mul.Tensor(sum_253, 7.086167800453515e-05)
    mul_2535: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_2536: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2534, mul_2535);  mul_2534 = mul_2535 = None
    unsqueeze_2310: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2536, 0);  mul_2536 = None
    unsqueeze_2311: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, 2);  unsqueeze_2310 = None
    unsqueeze_2312: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2311, 3);  unsqueeze_2311 = None
    mul_2537: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_302);  primals_302 = None
    unsqueeze_2313: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2537, 0);  mul_2537 = None
    unsqueeze_2314: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2313, 2);  unsqueeze_2313 = None
    unsqueeze_2315: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2314, 3);  unsqueeze_2314 = None
    mul_2538: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_701, unsqueeze_2312);  sub_701 = unsqueeze_2312 = None
    sub_703: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_125, mul_2538);  where_125 = mul_2538 = None
    sub_704: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_703, unsqueeze_2309);  sub_703 = unsqueeze_2309 = None
    mul_2539: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_704, unsqueeze_2315);  sub_704 = unsqueeze_2315 = None
    mul_2540: "f32[216]" = torch.ops.aten.mul.Tensor(sum_253, squeeze_226);  sum_253 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_234 = torch.ops.aten.convolution_backward.default(mul_2539, convolution_137, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2539 = convolution_137 = primals_301 = None
    getitem_1188: "f32[8, 216, 42, 42]" = convolution_backward_234[0]
    getitem_1189: "f32[216, 216, 1, 1]" = convolution_backward_234[1];  convolution_backward_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_235 = torch.ops.aten.convolution_backward.default(getitem_1188, relu_73, primals_300, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1188 = relu_73 = primals_300 = None
    getitem_1191: "f32[8, 216, 42, 42]" = convolution_backward_235[0]
    getitem_1192: "f32[216, 1, 5, 5]" = convolution_backward_235[1];  convolution_backward_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_126: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_116, full_default, getitem_1191);  le_116 = getitem_1191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1155: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1154, where_126);  add_1154 = where_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_254: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1153, [0, 2, 3])
    sub_705: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_2318);  convolution_136 = unsqueeze_2318 = None
    mul_2541: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1153, sub_705)
    sum_255: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2541, [0, 2, 3]);  mul_2541 = None
    mul_2542: "f32[216]" = torch.ops.aten.mul.Tensor(sum_254, 7.086167800453515e-05)
    unsqueeze_2319: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2542, 0);  mul_2542 = None
    unsqueeze_2320: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2319, 2);  unsqueeze_2319 = None
    unsqueeze_2321: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2320, 3);  unsqueeze_2320 = None
    mul_2543: "f32[216]" = torch.ops.aten.mul.Tensor(sum_255, 7.086167800453515e-05)
    mul_2544: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_2545: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2543, mul_2544);  mul_2543 = mul_2544 = None
    unsqueeze_2322: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2545, 0);  mul_2545 = None
    unsqueeze_2323: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, 2);  unsqueeze_2322 = None
    unsqueeze_2324: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2323, 3);  unsqueeze_2323 = None
    mul_2546: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_298);  primals_298 = None
    unsqueeze_2325: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2546, 0);  mul_2546 = None
    unsqueeze_2326: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2325, 2);  unsqueeze_2325 = None
    unsqueeze_2327: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2326, 3);  unsqueeze_2326 = None
    mul_2547: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_705, unsqueeze_2324);  sub_705 = unsqueeze_2324 = None
    sub_707: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1153, mul_2547);  add_1153 = mul_2547 = None
    sub_708: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_707, unsqueeze_2321);  sub_707 = unsqueeze_2321 = None
    mul_2548: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_708, unsqueeze_2327);  sub_708 = unsqueeze_2327 = None
    mul_2549: "f32[216]" = torch.ops.aten.mul.Tensor(sum_255, squeeze_223);  sum_255 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_236 = torch.ops.aten.convolution_backward.default(mul_2548, relu_72, primals_297, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2548 = relu_72 = primals_297 = None
    getitem_1194: "f32[8, 1080, 42, 42]" = convolution_backward_236[0]
    getitem_1195: "f32[216, 1080, 1, 1]" = convolution_backward_236[1];  convolution_backward_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_127: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_114, full_default, getitem_1194);  le_114 = getitem_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1156: "f32[8, 1080, 42, 42]" = torch.ops.aten.add.Tensor(where_114, where_127);  where_114 = where_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_256: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1155, [0, 2, 3])
    sub_709: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_2330);  convolution_135 = unsqueeze_2330 = None
    mul_2550: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1155, sub_709)
    sum_257: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2550, [0, 2, 3]);  mul_2550 = None
    mul_2551: "f32[216]" = torch.ops.aten.mul.Tensor(sum_256, 7.086167800453515e-05)
    unsqueeze_2331: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2551, 0);  mul_2551 = None
    unsqueeze_2332: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2331, 2);  unsqueeze_2331 = None
    unsqueeze_2333: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2332, 3);  unsqueeze_2332 = None
    mul_2552: "f32[216]" = torch.ops.aten.mul.Tensor(sum_257, 7.086167800453515e-05)
    mul_2553: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_2554: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2552, mul_2553);  mul_2552 = mul_2553 = None
    unsqueeze_2334: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2554, 0);  mul_2554 = None
    unsqueeze_2335: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, 2);  unsqueeze_2334 = None
    unsqueeze_2336: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2335, 3);  unsqueeze_2335 = None
    mul_2555: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_295);  primals_295 = None
    unsqueeze_2337: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2555, 0);  mul_2555 = None
    unsqueeze_2338: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2337, 2);  unsqueeze_2337 = None
    unsqueeze_2339: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2338, 3);  unsqueeze_2338 = None
    mul_2556: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_709, unsqueeze_2336);  sub_709 = unsqueeze_2336 = None
    sub_711: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1155, mul_2556);  add_1155 = mul_2556 = None
    sub_712: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_711, unsqueeze_2333);  sub_711 = unsqueeze_2333 = None
    mul_2557: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_712, unsqueeze_2339);  sub_712 = unsqueeze_2339 = None
    mul_2558: "f32[216]" = torch.ops.aten.mul.Tensor(sum_257, squeeze_220);  sum_257 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_237 = torch.ops.aten.convolution_backward.default(mul_2557, relu_58, primals_294, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2557 = primals_294 = None
    getitem_1197: "f32[8, 1080, 42, 42]" = convolution_backward_237[0]
    getitem_1198: "f32[216, 1080, 1, 1]" = convolution_backward_237[1];  convolution_backward_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_128: "b8[8, 1080, 42, 42]" = torch.ops.aten.le.Scalar(relu_58, 0)
    where_128: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_128, full_default, getitem_1197);  getitem_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_50: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1156, 1, 0, 216)
    slice_51: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1156, 1, 216, 432)
    slice_52: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1156, 1, 432, 648)
    slice_53: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1156, 1, 648, 864)
    slice_54: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1156, 1, 864, 1080);  add_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_258: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_54, [0, 2, 3])
    sub_713: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_2342);  convolution_134 = unsqueeze_2342 = None
    mul_2559: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_54, sub_713)
    sum_259: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2559, [0, 2, 3]);  mul_2559 = None
    mul_2560: "f32[216]" = torch.ops.aten.mul.Tensor(sum_258, 7.086167800453515e-05)
    unsqueeze_2343: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2560, 0);  mul_2560 = None
    unsqueeze_2344: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2343, 2);  unsqueeze_2343 = None
    unsqueeze_2345: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2344, 3);  unsqueeze_2344 = None
    mul_2561: "f32[216]" = torch.ops.aten.mul.Tensor(sum_259, 7.086167800453515e-05)
    mul_2562: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_2563: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2561, mul_2562);  mul_2561 = mul_2562 = None
    unsqueeze_2346: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2563, 0);  mul_2563 = None
    unsqueeze_2347: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, 2);  unsqueeze_2346 = None
    unsqueeze_2348: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2347, 3);  unsqueeze_2347 = None
    mul_2564: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_292);  primals_292 = None
    unsqueeze_2349: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2564, 0);  mul_2564 = None
    unsqueeze_2350: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2349, 2);  unsqueeze_2349 = None
    unsqueeze_2351: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2350, 3);  unsqueeze_2350 = None
    mul_2565: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_713, unsqueeze_2348);  sub_713 = unsqueeze_2348 = None
    sub_715: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_54, mul_2565);  mul_2565 = None
    sub_716: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_715, unsqueeze_2345);  sub_715 = unsqueeze_2345 = None
    mul_2566: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_716, unsqueeze_2351);  sub_716 = unsqueeze_2351 = None
    mul_2567: "f32[216]" = torch.ops.aten.mul.Tensor(sum_259, squeeze_217);  sum_259 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_238 = torch.ops.aten.convolution_backward.default(mul_2566, convolution_133, primals_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2566 = convolution_133 = primals_291 = None
    getitem_1200: "f32[8, 216, 42, 42]" = convolution_backward_238[0]
    getitem_1201: "f32[216, 216, 1, 1]" = convolution_backward_238[1];  convolution_backward_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_239 = torch.ops.aten.convolution_backward.default(getitem_1200, relu_70, primals_290, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1200 = primals_290 = None
    getitem_1203: "f32[8, 216, 42, 42]" = convolution_backward_239[0]
    getitem_1204: "f32[216, 1, 3, 3]" = convolution_backward_239[1];  convolution_backward_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_129: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_129: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_129, full_default, getitem_1203);  le_129 = getitem_1203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_260: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_717: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_2354);  convolution_132 = unsqueeze_2354 = None
    mul_2568: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_129, sub_717)
    sum_261: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2568, [0, 2, 3]);  mul_2568 = None
    mul_2569: "f32[216]" = torch.ops.aten.mul.Tensor(sum_260, 7.086167800453515e-05)
    unsqueeze_2355: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2569, 0);  mul_2569 = None
    unsqueeze_2356: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2355, 2);  unsqueeze_2355 = None
    unsqueeze_2357: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2356, 3);  unsqueeze_2356 = None
    mul_2570: "f32[216]" = torch.ops.aten.mul.Tensor(sum_261, 7.086167800453515e-05)
    mul_2571: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_2572: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2570, mul_2571);  mul_2570 = mul_2571 = None
    unsqueeze_2358: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2572, 0);  mul_2572 = None
    unsqueeze_2359: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, 2);  unsqueeze_2358 = None
    unsqueeze_2360: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2359, 3);  unsqueeze_2359 = None
    mul_2573: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_288);  primals_288 = None
    unsqueeze_2361: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2573, 0);  mul_2573 = None
    unsqueeze_2362: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2361, 2);  unsqueeze_2361 = None
    unsqueeze_2363: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2362, 3);  unsqueeze_2362 = None
    mul_2574: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_717, unsqueeze_2360);  sub_717 = unsqueeze_2360 = None
    sub_719: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_129, mul_2574);  where_129 = mul_2574 = None
    sub_720: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_719, unsqueeze_2357);  sub_719 = unsqueeze_2357 = None
    mul_2575: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_720, unsqueeze_2363);  sub_720 = unsqueeze_2363 = None
    mul_2576: "f32[216]" = torch.ops.aten.mul.Tensor(sum_261, squeeze_214);  sum_261 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_240 = torch.ops.aten.convolution_backward.default(mul_2575, convolution_131, primals_287, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2575 = convolution_131 = primals_287 = None
    getitem_1206: "f32[8, 216, 42, 42]" = convolution_backward_240[0]
    getitem_1207: "f32[216, 216, 1, 1]" = convolution_backward_240[1];  convolution_backward_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_241 = torch.ops.aten.convolution_backward.default(getitem_1206, relu_59, primals_286, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1206 = primals_286 = None
    getitem_1209: "f32[8, 216, 42, 42]" = convolution_backward_241[0]
    getitem_1210: "f32[216, 1, 3, 3]" = convolution_backward_241[1];  convolution_backward_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_130: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_59, 0)
    where_130: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_130, full_default, getitem_1209);  getitem_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_27: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_53, add_324, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_157)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1157: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_54, max_pool2d_with_indices_backward_27);  slice_54 = max_pool2d_with_indices_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_262: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_53, [0, 2, 3])
    sub_721: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_2366);  convolution_130 = unsqueeze_2366 = None
    mul_2577: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_53, sub_721)
    sum_263: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2577, [0, 2, 3]);  mul_2577 = None
    mul_2578: "f32[216]" = torch.ops.aten.mul.Tensor(sum_262, 7.086167800453515e-05)
    unsqueeze_2367: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2578, 0);  mul_2578 = None
    unsqueeze_2368: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2367, 2);  unsqueeze_2367 = None
    unsqueeze_2369: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2368, 3);  unsqueeze_2368 = None
    mul_2579: "f32[216]" = torch.ops.aten.mul.Tensor(sum_263, 7.086167800453515e-05)
    mul_2580: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_2581: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2579, mul_2580);  mul_2579 = mul_2580 = None
    unsqueeze_2370: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2581, 0);  mul_2581 = None
    unsqueeze_2371: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, 2);  unsqueeze_2370 = None
    unsqueeze_2372: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2371, 3);  unsqueeze_2371 = None
    mul_2582: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_284);  primals_284 = None
    unsqueeze_2373: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2582, 0);  mul_2582 = None
    unsqueeze_2374: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2373, 2);  unsqueeze_2373 = None
    unsqueeze_2375: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2374, 3);  unsqueeze_2374 = None
    mul_2583: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_721, unsqueeze_2372);  sub_721 = unsqueeze_2372 = None
    sub_723: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_53, mul_2583);  slice_53 = mul_2583 = None
    sub_724: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_723, unsqueeze_2369);  sub_723 = unsqueeze_2369 = None
    mul_2584: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_724, unsqueeze_2375);  sub_724 = unsqueeze_2375 = None
    mul_2585: "f32[216]" = torch.ops.aten.mul.Tensor(sum_263, squeeze_211);  sum_263 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_242 = torch.ops.aten.convolution_backward.default(mul_2584, convolution_129, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2584 = convolution_129 = primals_283 = None
    getitem_1212: "f32[8, 216, 42, 42]" = convolution_backward_242[0]
    getitem_1213: "f32[216, 216, 1, 1]" = convolution_backward_242[1];  convolution_backward_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_243 = torch.ops.aten.convolution_backward.default(getitem_1212, relu_68, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1212 = primals_282 = None
    getitem_1215: "f32[8, 216, 42, 42]" = convolution_backward_243[0]
    getitem_1216: "f32[216, 1, 3, 3]" = convolution_backward_243[1];  convolution_backward_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_131: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    where_131: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_131, full_default, getitem_1215);  le_131 = getitem_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_264: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_725: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_2378);  convolution_128 = unsqueeze_2378 = None
    mul_2586: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_131, sub_725)
    sum_265: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2586, [0, 2, 3]);  mul_2586 = None
    mul_2587: "f32[216]" = torch.ops.aten.mul.Tensor(sum_264, 7.086167800453515e-05)
    unsqueeze_2379: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2587, 0);  mul_2587 = None
    unsqueeze_2380: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2379, 2);  unsqueeze_2379 = None
    unsqueeze_2381: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2380, 3);  unsqueeze_2380 = None
    mul_2588: "f32[216]" = torch.ops.aten.mul.Tensor(sum_265, 7.086167800453515e-05)
    mul_2589: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_2590: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2588, mul_2589);  mul_2588 = mul_2589 = None
    unsqueeze_2382: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2590, 0);  mul_2590 = None
    unsqueeze_2383: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, 2);  unsqueeze_2382 = None
    unsqueeze_2384: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2383, 3);  unsqueeze_2383 = None
    mul_2591: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_280);  primals_280 = None
    unsqueeze_2385: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2591, 0);  mul_2591 = None
    unsqueeze_2386: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2385, 2);  unsqueeze_2385 = None
    unsqueeze_2387: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2386, 3);  unsqueeze_2386 = None
    mul_2592: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_725, unsqueeze_2384);  sub_725 = unsqueeze_2384 = None
    sub_727: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_131, mul_2592);  where_131 = mul_2592 = None
    sub_728: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_727, unsqueeze_2381);  sub_727 = unsqueeze_2381 = None
    mul_2593: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_728, unsqueeze_2387);  sub_728 = unsqueeze_2387 = None
    mul_2594: "f32[216]" = torch.ops.aten.mul.Tensor(sum_265, squeeze_208);  sum_265 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_244 = torch.ops.aten.convolution_backward.default(mul_2593, convolution_127, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2593 = convolution_127 = primals_279 = None
    getitem_1218: "f32[8, 216, 42, 42]" = convolution_backward_244[0]
    getitem_1219: "f32[216, 216, 1, 1]" = convolution_backward_244[1];  convolution_backward_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_245 = torch.ops.aten.convolution_backward.default(getitem_1218, relu_67, primals_278, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1218 = primals_278 = None
    getitem_1221: "f32[8, 216, 42, 42]" = convolution_backward_245[0]
    getitem_1222: "f32[216, 1, 3, 3]" = convolution_backward_245[1];  convolution_backward_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_132: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
    where_132: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_132, full_default, getitem_1221);  le_132 = getitem_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1158: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_52, where_132);  slice_52 = where_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_266: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1158, [0, 2, 3])
    sub_729: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_2390);  convolution_126 = unsqueeze_2390 = None
    mul_2595: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1158, sub_729)
    sum_267: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2595, [0, 2, 3]);  mul_2595 = None
    mul_2596: "f32[216]" = torch.ops.aten.mul.Tensor(sum_266, 7.086167800453515e-05)
    unsqueeze_2391: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2596, 0);  mul_2596 = None
    unsqueeze_2392: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2391, 2);  unsqueeze_2391 = None
    unsqueeze_2393: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2392, 3);  unsqueeze_2392 = None
    mul_2597: "f32[216]" = torch.ops.aten.mul.Tensor(sum_267, 7.086167800453515e-05)
    mul_2598: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_2599: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2597, mul_2598);  mul_2597 = mul_2598 = None
    unsqueeze_2394: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2599, 0);  mul_2599 = None
    unsqueeze_2395: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2394, 2);  unsqueeze_2394 = None
    unsqueeze_2396: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2395, 3);  unsqueeze_2395 = None
    mul_2600: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_276);  primals_276 = None
    unsqueeze_2397: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2600, 0);  mul_2600 = None
    unsqueeze_2398: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2397, 2);  unsqueeze_2397 = None
    unsqueeze_2399: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2398, 3);  unsqueeze_2398 = None
    mul_2601: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_729, unsqueeze_2396);  sub_729 = unsqueeze_2396 = None
    sub_731: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1158, mul_2601);  mul_2601 = None
    sub_732: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_731, unsqueeze_2393);  sub_731 = None
    mul_2602: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_732, unsqueeze_2399);  sub_732 = unsqueeze_2399 = None
    mul_2603: "f32[216]" = torch.ops.aten.mul.Tensor(sum_267, squeeze_205);  sum_267 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_246 = torch.ops.aten.convolution_backward.default(mul_2602, convolution_125, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2602 = convolution_125 = primals_275 = None
    getitem_1224: "f32[8, 216, 42, 42]" = convolution_backward_246[0]
    getitem_1225: "f32[216, 216, 1, 1]" = convolution_backward_246[1];  convolution_backward_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_247 = torch.ops.aten.convolution_backward.default(getitem_1224, relu_66, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1224 = primals_274 = None
    getitem_1227: "f32[8, 216, 42, 42]" = convolution_backward_247[0]
    getitem_1228: "f32[216, 1, 3, 3]" = convolution_backward_247[1];  convolution_backward_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_133: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
    where_133: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_133, full_default, getitem_1227);  le_133 = getitem_1227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_268: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_733: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_2402);  convolution_124 = unsqueeze_2402 = None
    mul_2604: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_133, sub_733)
    sum_269: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2604, [0, 2, 3]);  mul_2604 = None
    mul_2605: "f32[216]" = torch.ops.aten.mul.Tensor(sum_268, 7.086167800453515e-05)
    unsqueeze_2403: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2605, 0);  mul_2605 = None
    unsqueeze_2404: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2403, 2);  unsqueeze_2403 = None
    unsqueeze_2405: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2404, 3);  unsqueeze_2404 = None
    mul_2606: "f32[216]" = torch.ops.aten.mul.Tensor(sum_269, 7.086167800453515e-05)
    mul_2607: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_2608: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2606, mul_2607);  mul_2606 = mul_2607 = None
    unsqueeze_2406: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2608, 0);  mul_2608 = None
    unsqueeze_2407: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2406, 2);  unsqueeze_2406 = None
    unsqueeze_2408: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2407, 3);  unsqueeze_2407 = None
    mul_2609: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_272);  primals_272 = None
    unsqueeze_2409: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2609, 0);  mul_2609 = None
    unsqueeze_2410: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2409, 2);  unsqueeze_2409 = None
    unsqueeze_2411: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2410, 3);  unsqueeze_2410 = None
    mul_2610: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_733, unsqueeze_2408);  sub_733 = unsqueeze_2408 = None
    sub_735: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_133, mul_2610);  where_133 = mul_2610 = None
    sub_736: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_735, unsqueeze_2405);  sub_735 = unsqueeze_2405 = None
    mul_2611: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_736, unsqueeze_2411);  sub_736 = unsqueeze_2411 = None
    mul_2612: "f32[216]" = torch.ops.aten.mul.Tensor(sum_269, squeeze_202);  sum_269 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_248 = torch.ops.aten.convolution_backward.default(mul_2611, convolution_123, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2611 = convolution_123 = primals_271 = None
    getitem_1230: "f32[8, 216, 42, 42]" = convolution_backward_248[0]
    getitem_1231: "f32[216, 216, 1, 1]" = convolution_backward_248[1];  convolution_backward_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_249 = torch.ops.aten.convolution_backward.default(getitem_1230, relu_61, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1230 = primals_270 = None
    getitem_1233: "f32[8, 216, 42, 42]" = convolution_backward_249[0]
    getitem_1234: "f32[216, 1, 3, 3]" = convolution_backward_249[1];  convolution_backward_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_134: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_61, 0)
    where_134: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_134, full_default, getitem_1233);  getitem_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1159: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1157, where_134);  add_1157 = where_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_737: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_2414);  convolution_122 = unsqueeze_2414 = None
    mul_2613: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1158, sub_737)
    sum_271: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2613, [0, 2, 3]);  mul_2613 = None
    mul_2615: "f32[216]" = torch.ops.aten.mul.Tensor(sum_271, 7.086167800453515e-05)
    mul_2616: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_2617: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2615, mul_2616);  mul_2615 = mul_2616 = None
    unsqueeze_2418: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2617, 0);  mul_2617 = None
    unsqueeze_2419: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2418, 2);  unsqueeze_2418 = None
    unsqueeze_2420: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2419, 3);  unsqueeze_2419 = None
    mul_2618: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_268);  primals_268 = None
    unsqueeze_2421: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2618, 0);  mul_2618 = None
    unsqueeze_2422: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2421, 2);  unsqueeze_2421 = None
    unsqueeze_2423: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2422, 3);  unsqueeze_2422 = None
    mul_2619: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_737, unsqueeze_2420);  sub_737 = unsqueeze_2420 = None
    sub_739: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1158, mul_2619);  add_1158 = mul_2619 = None
    sub_740: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_739, unsqueeze_2393);  sub_739 = unsqueeze_2393 = None
    mul_2620: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_740, unsqueeze_2423);  sub_740 = unsqueeze_2423 = None
    mul_2621: "f32[216]" = torch.ops.aten.mul.Tensor(sum_271, squeeze_199);  sum_271 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_250 = torch.ops.aten.convolution_backward.default(mul_2620, convolution_121, primals_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2620 = convolution_121 = primals_267 = None
    getitem_1236: "f32[8, 216, 42, 42]" = convolution_backward_250[0]
    getitem_1237: "f32[216, 216, 1, 1]" = convolution_backward_250[1];  convolution_backward_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_251 = torch.ops.aten.convolution_backward.default(getitem_1236, relu_64, primals_266, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1236 = primals_266 = None
    getitem_1239: "f32[8, 216, 42, 42]" = convolution_backward_251[0]
    getitem_1240: "f32[216, 1, 5, 5]" = convolution_backward_251[1];  convolution_backward_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_135: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
    where_135: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_135, full_default, getitem_1239);  le_135 = getitem_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_272: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_135, [0, 2, 3])
    sub_741: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_2426);  convolution_120 = unsqueeze_2426 = None
    mul_2622: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_135, sub_741)
    sum_273: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2622, [0, 2, 3]);  mul_2622 = None
    mul_2623: "f32[216]" = torch.ops.aten.mul.Tensor(sum_272, 7.086167800453515e-05)
    unsqueeze_2427: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2623, 0);  mul_2623 = None
    unsqueeze_2428: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2427, 2);  unsqueeze_2427 = None
    unsqueeze_2429: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2428, 3);  unsqueeze_2428 = None
    mul_2624: "f32[216]" = torch.ops.aten.mul.Tensor(sum_273, 7.086167800453515e-05)
    mul_2625: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_2626: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2624, mul_2625);  mul_2624 = mul_2625 = None
    unsqueeze_2430: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2626, 0);  mul_2626 = None
    unsqueeze_2431: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2430, 2);  unsqueeze_2430 = None
    unsqueeze_2432: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2431, 3);  unsqueeze_2431 = None
    mul_2627: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_264);  primals_264 = None
    unsqueeze_2433: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2627, 0);  mul_2627 = None
    unsqueeze_2434: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2433, 2);  unsqueeze_2433 = None
    unsqueeze_2435: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2434, 3);  unsqueeze_2434 = None
    mul_2628: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_741, unsqueeze_2432);  sub_741 = unsqueeze_2432 = None
    sub_743: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_135, mul_2628);  where_135 = mul_2628 = None
    sub_744: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_743, unsqueeze_2429);  sub_743 = unsqueeze_2429 = None
    mul_2629: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_744, unsqueeze_2435);  sub_744 = unsqueeze_2435 = None
    mul_2630: "f32[216]" = torch.ops.aten.mul.Tensor(sum_273, squeeze_196);  sum_273 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_252 = torch.ops.aten.convolution_backward.default(mul_2629, convolution_119, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2629 = convolution_119 = primals_263 = None
    getitem_1242: "f32[8, 216, 42, 42]" = convolution_backward_252[0]
    getitem_1243: "f32[216, 216, 1, 1]" = convolution_backward_252[1];  convolution_backward_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_253 = torch.ops.aten.convolution_backward.default(getitem_1242, relu_61, primals_262, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1242 = primals_262 = None
    getitem_1245: "f32[8, 216, 42, 42]" = convolution_backward_253[0]
    getitem_1246: "f32[216, 1, 5, 5]" = convolution_backward_253[1];  convolution_backward_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_136: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_134, full_default, getitem_1245);  getitem_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1160: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1159, where_136);  add_1159 = where_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_28: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_51, add_324, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_157);  add_324 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1161: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1160, max_pool2d_with_indices_backward_28);  add_1160 = max_pool2d_with_indices_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_274: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_51, [0, 2, 3])
    sub_745: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_2438);  convolution_118 = unsqueeze_2438 = None
    mul_2631: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_51, sub_745)
    sum_275: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2631, [0, 2, 3]);  mul_2631 = None
    mul_2632: "f32[216]" = torch.ops.aten.mul.Tensor(sum_274, 7.086167800453515e-05)
    unsqueeze_2439: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2632, 0);  mul_2632 = None
    unsqueeze_2440: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2439, 2);  unsqueeze_2439 = None
    unsqueeze_2441: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2440, 3);  unsqueeze_2440 = None
    mul_2633: "f32[216]" = torch.ops.aten.mul.Tensor(sum_275, 7.086167800453515e-05)
    mul_2634: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_2635: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2633, mul_2634);  mul_2633 = mul_2634 = None
    unsqueeze_2442: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2635, 0);  mul_2635 = None
    unsqueeze_2443: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2442, 2);  unsqueeze_2442 = None
    unsqueeze_2444: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2443, 3);  unsqueeze_2443 = None
    mul_2636: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_260);  primals_260 = None
    unsqueeze_2445: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2636, 0);  mul_2636 = None
    unsqueeze_2446: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2445, 2);  unsqueeze_2445 = None
    unsqueeze_2447: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2446, 3);  unsqueeze_2446 = None
    mul_2637: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_745, unsqueeze_2444);  sub_745 = unsqueeze_2444 = None
    sub_747: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_51, mul_2637);  slice_51 = mul_2637 = None
    sub_748: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_747, unsqueeze_2441);  sub_747 = unsqueeze_2441 = None
    mul_2638: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_748, unsqueeze_2447);  sub_748 = unsqueeze_2447 = None
    mul_2639: "f32[216]" = torch.ops.aten.mul.Tensor(sum_275, squeeze_193);  sum_275 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_254 = torch.ops.aten.convolution_backward.default(mul_2638, convolution_117, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2638 = convolution_117 = primals_259 = None
    getitem_1248: "f32[8, 216, 42, 42]" = convolution_backward_254[0]
    getitem_1249: "f32[216, 216, 1, 1]" = convolution_backward_254[1];  convolution_backward_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_255 = torch.ops.aten.convolution_backward.default(getitem_1248, relu_62, primals_258, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1248 = primals_258 = None
    getitem_1251: "f32[8, 216, 42, 42]" = convolution_backward_255[0]
    getitem_1252: "f32[216, 1, 7, 7]" = convolution_backward_255[1];  convolution_backward_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_137: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_137: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_137, full_default, getitem_1251);  le_137 = getitem_1251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_276: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_137, [0, 2, 3])
    sub_749: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_2450);  convolution_116 = unsqueeze_2450 = None
    mul_2640: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_137, sub_749)
    sum_277: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2640, [0, 2, 3]);  mul_2640 = None
    mul_2641: "f32[216]" = torch.ops.aten.mul.Tensor(sum_276, 7.086167800453515e-05)
    unsqueeze_2451: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2641, 0);  mul_2641 = None
    unsqueeze_2452: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2451, 2);  unsqueeze_2451 = None
    unsqueeze_2453: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2452, 3);  unsqueeze_2452 = None
    mul_2642: "f32[216]" = torch.ops.aten.mul.Tensor(sum_277, 7.086167800453515e-05)
    mul_2643: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_2644: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2642, mul_2643);  mul_2642 = mul_2643 = None
    unsqueeze_2454: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2644, 0);  mul_2644 = None
    unsqueeze_2455: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2454, 2);  unsqueeze_2454 = None
    unsqueeze_2456: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2455, 3);  unsqueeze_2455 = None
    mul_2645: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_256);  primals_256 = None
    unsqueeze_2457: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2645, 0);  mul_2645 = None
    unsqueeze_2458: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2457, 2);  unsqueeze_2457 = None
    unsqueeze_2459: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2458, 3);  unsqueeze_2458 = None
    mul_2646: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_749, unsqueeze_2456);  sub_749 = unsqueeze_2456 = None
    sub_751: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_137, mul_2646);  where_137 = mul_2646 = None
    sub_752: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_751, unsqueeze_2453);  sub_751 = unsqueeze_2453 = None
    mul_2647: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_752, unsqueeze_2459);  sub_752 = unsqueeze_2459 = None
    mul_2648: "f32[216]" = torch.ops.aten.mul.Tensor(sum_277, squeeze_190);  sum_277 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_256 = torch.ops.aten.convolution_backward.default(mul_2647, convolution_115, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2647 = convolution_115 = primals_255 = None
    getitem_1254: "f32[8, 216, 42, 42]" = convolution_backward_256[0]
    getitem_1255: "f32[216, 216, 1, 1]" = convolution_backward_256[1];  convolution_backward_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_257 = torch.ops.aten.convolution_backward.default(getitem_1254, relu_61, primals_254, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1254 = relu_61 = primals_254 = None
    getitem_1257: "f32[8, 216, 42, 42]" = convolution_backward_257[0]
    getitem_1258: "f32[216, 1, 7, 7]" = convolution_backward_257[1];  convolution_backward_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_138: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_134, full_default, getitem_1257);  le_134 = getitem_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1162: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1161, where_138);  add_1161 = where_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_29: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_50, add_319, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_151);  add_319 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1163: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(where_130, max_pool2d_with_indices_backward_29);  where_130 = max_pool2d_with_indices_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_278: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_50, [0, 2, 3])
    sub_753: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_2462);  convolution_114 = unsqueeze_2462 = None
    mul_2649: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_50, sub_753)
    sum_279: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2649, [0, 2, 3]);  mul_2649 = None
    mul_2650: "f32[216]" = torch.ops.aten.mul.Tensor(sum_278, 7.086167800453515e-05)
    unsqueeze_2463: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2650, 0);  mul_2650 = None
    unsqueeze_2464: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2463, 2);  unsqueeze_2463 = None
    unsqueeze_2465: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2464, 3);  unsqueeze_2464 = None
    mul_2651: "f32[216]" = torch.ops.aten.mul.Tensor(sum_279, 7.086167800453515e-05)
    mul_2652: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_2653: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2651, mul_2652);  mul_2651 = mul_2652 = None
    unsqueeze_2466: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2653, 0);  mul_2653 = None
    unsqueeze_2467: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2466, 2);  unsqueeze_2466 = None
    unsqueeze_2468: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2467, 3);  unsqueeze_2467 = None
    mul_2654: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_252);  primals_252 = None
    unsqueeze_2469: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2654, 0);  mul_2654 = None
    unsqueeze_2470: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2469, 2);  unsqueeze_2469 = None
    unsqueeze_2471: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2470, 3);  unsqueeze_2470 = None
    mul_2655: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_753, unsqueeze_2468);  sub_753 = unsqueeze_2468 = None
    sub_755: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_50, mul_2655);  slice_50 = mul_2655 = None
    sub_756: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_755, unsqueeze_2465);  sub_755 = unsqueeze_2465 = None
    mul_2656: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_756, unsqueeze_2471);  sub_756 = unsqueeze_2471 = None
    mul_2657: "f32[216]" = torch.ops.aten.mul.Tensor(sum_279, squeeze_187);  sum_279 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_258 = torch.ops.aten.convolution_backward.default(mul_2656, convolution_113, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2656 = convolution_113 = primals_251 = None
    getitem_1260: "f32[8, 216, 42, 42]" = convolution_backward_258[0]
    getitem_1261: "f32[216, 216, 1, 1]" = convolution_backward_258[1];  convolution_backward_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_259 = torch.ops.aten.convolution_backward.default(getitem_1260, relu_60, primals_250, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1260 = primals_250 = None
    getitem_1263: "f32[8, 216, 42, 42]" = convolution_backward_259[0]
    getitem_1264: "f32[216, 1, 5, 5]" = convolution_backward_259[1];  convolution_backward_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_139: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_139: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_139, full_default, getitem_1263);  le_139 = getitem_1263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_280: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_139, [0, 2, 3])
    sub_757: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_2474);  convolution_112 = unsqueeze_2474 = None
    mul_2658: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_139, sub_757)
    sum_281: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2658, [0, 2, 3]);  mul_2658 = None
    mul_2659: "f32[216]" = torch.ops.aten.mul.Tensor(sum_280, 7.086167800453515e-05)
    unsqueeze_2475: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2659, 0);  mul_2659 = None
    unsqueeze_2476: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2475, 2);  unsqueeze_2475 = None
    unsqueeze_2477: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2476, 3);  unsqueeze_2476 = None
    mul_2660: "f32[216]" = torch.ops.aten.mul.Tensor(sum_281, 7.086167800453515e-05)
    mul_2661: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_2662: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2660, mul_2661);  mul_2660 = mul_2661 = None
    unsqueeze_2478: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2662, 0);  mul_2662 = None
    unsqueeze_2479: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2478, 2);  unsqueeze_2478 = None
    unsqueeze_2480: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2479, 3);  unsqueeze_2479 = None
    mul_2663: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_248);  primals_248 = None
    unsqueeze_2481: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2663, 0);  mul_2663 = None
    unsqueeze_2482: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2481, 2);  unsqueeze_2481 = None
    unsqueeze_2483: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2482, 3);  unsqueeze_2482 = None
    mul_2664: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_757, unsqueeze_2480);  sub_757 = unsqueeze_2480 = None
    sub_759: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_139, mul_2664);  where_139 = mul_2664 = None
    sub_760: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_759, unsqueeze_2477);  sub_759 = unsqueeze_2477 = None
    mul_2665: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_760, unsqueeze_2483);  sub_760 = unsqueeze_2483 = None
    mul_2666: "f32[216]" = torch.ops.aten.mul.Tensor(sum_281, squeeze_184);  sum_281 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_260 = torch.ops.aten.convolution_backward.default(mul_2665, convolution_111, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2665 = convolution_111 = primals_247 = None
    getitem_1266: "f32[8, 216, 42, 42]" = convolution_backward_260[0]
    getitem_1267: "f32[216, 216, 1, 1]" = convolution_backward_260[1];  convolution_backward_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_261 = torch.ops.aten.convolution_backward.default(getitem_1266, relu_59, primals_246, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1266 = relu_59 = primals_246 = None
    getitem_1269: "f32[8, 216, 42, 42]" = convolution_backward_261[0]
    getitem_1270: "f32[216, 1, 5, 5]" = convolution_backward_261[1];  convolution_backward_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_140: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_130, full_default, getitem_1269);  le_130 = getitem_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1164: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1163, where_140);  add_1163 = where_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_282: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1162, [0, 2, 3])
    sub_761: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_2486);  convolution_110 = unsqueeze_2486 = None
    mul_2667: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1162, sub_761)
    sum_283: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2667, [0, 2, 3]);  mul_2667 = None
    mul_2668: "f32[216]" = torch.ops.aten.mul.Tensor(sum_282, 7.086167800453515e-05)
    unsqueeze_2487: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2668, 0);  mul_2668 = None
    unsqueeze_2488: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2487, 2);  unsqueeze_2487 = None
    unsqueeze_2489: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2488, 3);  unsqueeze_2488 = None
    mul_2669: "f32[216]" = torch.ops.aten.mul.Tensor(sum_283, 7.086167800453515e-05)
    mul_2670: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_2671: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2669, mul_2670);  mul_2669 = mul_2670 = None
    unsqueeze_2490: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2671, 0);  mul_2671 = None
    unsqueeze_2491: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2490, 2);  unsqueeze_2490 = None
    unsqueeze_2492: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2491, 3);  unsqueeze_2491 = None
    mul_2672: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_244);  primals_244 = None
    unsqueeze_2493: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2672, 0);  mul_2672 = None
    unsqueeze_2494: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2493, 2);  unsqueeze_2493 = None
    unsqueeze_2495: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2494, 3);  unsqueeze_2494 = None
    mul_2673: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_761, unsqueeze_2492);  sub_761 = unsqueeze_2492 = None
    sub_763: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1162, mul_2673);  add_1162 = mul_2673 = None
    sub_764: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_763, unsqueeze_2489);  sub_763 = unsqueeze_2489 = None
    mul_2674: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_764, unsqueeze_2495);  sub_764 = unsqueeze_2495 = None
    mul_2675: "f32[216]" = torch.ops.aten.mul.Tensor(sum_283, squeeze_181);  sum_283 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_262 = torch.ops.aten.convolution_backward.default(mul_2674, relu_58, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2674 = relu_58 = primals_243 = None
    getitem_1272: "f32[8, 1080, 42, 42]" = convolution_backward_262[0]
    getitem_1273: "f32[216, 1080, 1, 1]" = convolution_backward_262[1];  convolution_backward_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_141: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_128, full_default, getitem_1272);  le_128 = getitem_1272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1165: "f32[8, 1080, 42, 42]" = torch.ops.aten.add.Tensor(where_128, where_141);  where_128 = where_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_284: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1164, [0, 2, 3])
    sub_765: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_2498);  convolution_109 = unsqueeze_2498 = None
    mul_2676: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1164, sub_765)
    sum_285: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2676, [0, 2, 3]);  mul_2676 = None
    mul_2677: "f32[216]" = torch.ops.aten.mul.Tensor(sum_284, 7.086167800453515e-05)
    unsqueeze_2499: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2677, 0);  mul_2677 = None
    unsqueeze_2500: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2499, 2);  unsqueeze_2499 = None
    unsqueeze_2501: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2500, 3);  unsqueeze_2500 = None
    mul_2678: "f32[216]" = torch.ops.aten.mul.Tensor(sum_285, 7.086167800453515e-05)
    mul_2679: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_2680: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2678, mul_2679);  mul_2678 = mul_2679 = None
    unsqueeze_2502: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2680, 0);  mul_2680 = None
    unsqueeze_2503: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2502, 2);  unsqueeze_2502 = None
    unsqueeze_2504: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2503, 3);  unsqueeze_2503 = None
    mul_2681: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_241);  primals_241 = None
    unsqueeze_2505: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2681, 0);  mul_2681 = None
    unsqueeze_2506: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2505, 2);  unsqueeze_2505 = None
    unsqueeze_2507: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2506, 3);  unsqueeze_2506 = None
    mul_2682: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_765, unsqueeze_2504);  sub_765 = unsqueeze_2504 = None
    sub_767: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1164, mul_2682);  add_1164 = mul_2682 = None
    sub_768: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_767, unsqueeze_2501);  sub_767 = unsqueeze_2501 = None
    mul_2683: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_768, unsqueeze_2507);  sub_768 = unsqueeze_2507 = None
    mul_2684: "f32[216]" = torch.ops.aten.mul.Tensor(sum_285, squeeze_178);  sum_285 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_263 = torch.ops.aten.convolution_backward.default(mul_2683, relu_44, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2683 = primals_240 = None
    getitem_1275: "f32[8, 1080, 42, 42]" = convolution_backward_263[0]
    getitem_1276: "f32[216, 1080, 1, 1]" = convolution_backward_263[1];  convolution_backward_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_142: "b8[8, 1080, 42, 42]" = torch.ops.aten.le.Scalar(relu_44, 0)
    where_142: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_142, full_default, getitem_1275);  getitem_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_55: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1165, 1, 0, 216)
    slice_56: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1165, 1, 216, 432)
    slice_57: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1165, 1, 432, 648)
    slice_58: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1165, 1, 648, 864)
    slice_59: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1165, 1, 864, 1080);  add_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_286: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_59, [0, 2, 3])
    sub_769: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_2510);  convolution_108 = unsqueeze_2510 = None
    mul_2685: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_59, sub_769)
    sum_287: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2685, [0, 2, 3]);  mul_2685 = None
    mul_2686: "f32[216]" = torch.ops.aten.mul.Tensor(sum_286, 7.086167800453515e-05)
    unsqueeze_2511: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2686, 0);  mul_2686 = None
    unsqueeze_2512: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2511, 2);  unsqueeze_2511 = None
    unsqueeze_2513: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2512, 3);  unsqueeze_2512 = None
    mul_2687: "f32[216]" = torch.ops.aten.mul.Tensor(sum_287, 7.086167800453515e-05)
    mul_2688: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_2689: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2687, mul_2688);  mul_2687 = mul_2688 = None
    unsqueeze_2514: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2689, 0);  mul_2689 = None
    unsqueeze_2515: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2514, 2);  unsqueeze_2514 = None
    unsqueeze_2516: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2515, 3);  unsqueeze_2515 = None
    mul_2690: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_238);  primals_238 = None
    unsqueeze_2517: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2690, 0);  mul_2690 = None
    unsqueeze_2518: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2517, 2);  unsqueeze_2517 = None
    unsqueeze_2519: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2518, 3);  unsqueeze_2518 = None
    mul_2691: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_769, unsqueeze_2516);  sub_769 = unsqueeze_2516 = None
    sub_771: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_59, mul_2691);  mul_2691 = None
    sub_772: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_771, unsqueeze_2513);  sub_771 = unsqueeze_2513 = None
    mul_2692: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_772, unsqueeze_2519);  sub_772 = unsqueeze_2519 = None
    mul_2693: "f32[216]" = torch.ops.aten.mul.Tensor(sum_287, squeeze_175);  sum_287 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_264 = torch.ops.aten.convolution_backward.default(mul_2692, convolution_107, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2692 = convolution_107 = primals_237 = None
    getitem_1278: "f32[8, 216, 42, 42]" = convolution_backward_264[0]
    getitem_1279: "f32[216, 216, 1, 1]" = convolution_backward_264[1];  convolution_backward_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_265 = torch.ops.aten.convolution_backward.default(getitem_1278, relu_56, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1278 = primals_236 = None
    getitem_1281: "f32[8, 216, 42, 42]" = convolution_backward_265[0]
    getitem_1282: "f32[216, 1, 3, 3]" = convolution_backward_265[1];  convolution_backward_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_143: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_143: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_143, full_default, getitem_1281);  le_143 = getitem_1281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_288: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_143, [0, 2, 3])
    sub_773: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_2522);  convolution_106 = unsqueeze_2522 = None
    mul_2694: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_143, sub_773)
    sum_289: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2694, [0, 2, 3]);  mul_2694 = None
    mul_2695: "f32[216]" = torch.ops.aten.mul.Tensor(sum_288, 7.086167800453515e-05)
    unsqueeze_2523: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2695, 0);  mul_2695 = None
    unsqueeze_2524: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2523, 2);  unsqueeze_2523 = None
    unsqueeze_2525: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2524, 3);  unsqueeze_2524 = None
    mul_2696: "f32[216]" = torch.ops.aten.mul.Tensor(sum_289, 7.086167800453515e-05)
    mul_2697: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_2698: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2696, mul_2697);  mul_2696 = mul_2697 = None
    unsqueeze_2526: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2698, 0);  mul_2698 = None
    unsqueeze_2527: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2526, 2);  unsqueeze_2526 = None
    unsqueeze_2528: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2527, 3);  unsqueeze_2527 = None
    mul_2699: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_234);  primals_234 = None
    unsqueeze_2529: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2699, 0);  mul_2699 = None
    unsqueeze_2530: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2529, 2);  unsqueeze_2529 = None
    unsqueeze_2531: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2530, 3);  unsqueeze_2530 = None
    mul_2700: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_773, unsqueeze_2528);  sub_773 = unsqueeze_2528 = None
    sub_775: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_143, mul_2700);  where_143 = mul_2700 = None
    sub_776: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_775, unsqueeze_2525);  sub_775 = unsqueeze_2525 = None
    mul_2701: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_776, unsqueeze_2531);  sub_776 = unsqueeze_2531 = None
    mul_2702: "f32[216]" = torch.ops.aten.mul.Tensor(sum_289, squeeze_172);  sum_289 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_266 = torch.ops.aten.convolution_backward.default(mul_2701, convolution_105, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2701 = convolution_105 = primals_233 = None
    getitem_1284: "f32[8, 216, 42, 42]" = convolution_backward_266[0]
    getitem_1285: "f32[216, 216, 1, 1]" = convolution_backward_266[1];  convolution_backward_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_267 = torch.ops.aten.convolution_backward.default(getitem_1284, relu_45, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1284 = primals_232 = None
    getitem_1287: "f32[8, 216, 42, 42]" = convolution_backward_267[0]
    getitem_1288: "f32[216, 1, 3, 3]" = convolution_backward_267[1];  convolution_backward_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_144: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_45, 0)
    where_144: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_144, full_default, getitem_1287);  getitem_1287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_30: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_58, add_249, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1166: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_59, max_pool2d_with_indices_backward_30);  slice_59 = max_pool2d_with_indices_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_290: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_58, [0, 2, 3])
    sub_777: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_2534);  convolution_104 = unsqueeze_2534 = None
    mul_2703: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_58, sub_777)
    sum_291: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2703, [0, 2, 3]);  mul_2703 = None
    mul_2704: "f32[216]" = torch.ops.aten.mul.Tensor(sum_290, 7.086167800453515e-05)
    unsqueeze_2535: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2704, 0);  mul_2704 = None
    unsqueeze_2536: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2535, 2);  unsqueeze_2535 = None
    unsqueeze_2537: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2536, 3);  unsqueeze_2536 = None
    mul_2705: "f32[216]" = torch.ops.aten.mul.Tensor(sum_291, 7.086167800453515e-05)
    mul_2706: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_2707: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2705, mul_2706);  mul_2705 = mul_2706 = None
    unsqueeze_2538: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2707, 0);  mul_2707 = None
    unsqueeze_2539: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2538, 2);  unsqueeze_2538 = None
    unsqueeze_2540: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2539, 3);  unsqueeze_2539 = None
    mul_2708: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_230);  primals_230 = None
    unsqueeze_2541: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2708, 0);  mul_2708 = None
    unsqueeze_2542: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2541, 2);  unsqueeze_2541 = None
    unsqueeze_2543: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2542, 3);  unsqueeze_2542 = None
    mul_2709: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_777, unsqueeze_2540);  sub_777 = unsqueeze_2540 = None
    sub_779: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_58, mul_2709);  slice_58 = mul_2709 = None
    sub_780: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_779, unsqueeze_2537);  sub_779 = unsqueeze_2537 = None
    mul_2710: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_780, unsqueeze_2543);  sub_780 = unsqueeze_2543 = None
    mul_2711: "f32[216]" = torch.ops.aten.mul.Tensor(sum_291, squeeze_169);  sum_291 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_268 = torch.ops.aten.convolution_backward.default(mul_2710, convolution_103, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2710 = convolution_103 = primals_229 = None
    getitem_1290: "f32[8, 216, 42, 42]" = convolution_backward_268[0]
    getitem_1291: "f32[216, 216, 1, 1]" = convolution_backward_268[1];  convolution_backward_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_269 = torch.ops.aten.convolution_backward.default(getitem_1290, relu_54, primals_228, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1290 = primals_228 = None
    getitem_1293: "f32[8, 216, 42, 42]" = convolution_backward_269[0]
    getitem_1294: "f32[216, 1, 3, 3]" = convolution_backward_269[1];  convolution_backward_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_145: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_145: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_145, full_default, getitem_1293);  le_145 = getitem_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_292: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_145, [0, 2, 3])
    sub_781: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_2546);  convolution_102 = unsqueeze_2546 = None
    mul_2712: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_145, sub_781)
    sum_293: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2712, [0, 2, 3]);  mul_2712 = None
    mul_2713: "f32[216]" = torch.ops.aten.mul.Tensor(sum_292, 7.086167800453515e-05)
    unsqueeze_2547: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2713, 0);  mul_2713 = None
    unsqueeze_2548: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2547, 2);  unsqueeze_2547 = None
    unsqueeze_2549: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2548, 3);  unsqueeze_2548 = None
    mul_2714: "f32[216]" = torch.ops.aten.mul.Tensor(sum_293, 7.086167800453515e-05)
    mul_2715: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_2716: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2714, mul_2715);  mul_2714 = mul_2715 = None
    unsqueeze_2550: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2716, 0);  mul_2716 = None
    unsqueeze_2551: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2550, 2);  unsqueeze_2550 = None
    unsqueeze_2552: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2551, 3);  unsqueeze_2551 = None
    mul_2717: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_226);  primals_226 = None
    unsqueeze_2553: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2717, 0);  mul_2717 = None
    unsqueeze_2554: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2553, 2);  unsqueeze_2553 = None
    unsqueeze_2555: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2554, 3);  unsqueeze_2554 = None
    mul_2718: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_781, unsqueeze_2552);  sub_781 = unsqueeze_2552 = None
    sub_783: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_145, mul_2718);  where_145 = mul_2718 = None
    sub_784: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_783, unsqueeze_2549);  sub_783 = unsqueeze_2549 = None
    mul_2719: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_784, unsqueeze_2555);  sub_784 = unsqueeze_2555 = None
    mul_2720: "f32[216]" = torch.ops.aten.mul.Tensor(sum_293, squeeze_166);  sum_293 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_270 = torch.ops.aten.convolution_backward.default(mul_2719, convolution_101, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2719 = convolution_101 = primals_225 = None
    getitem_1296: "f32[8, 216, 42, 42]" = convolution_backward_270[0]
    getitem_1297: "f32[216, 216, 1, 1]" = convolution_backward_270[1];  convolution_backward_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_271 = torch.ops.aten.convolution_backward.default(getitem_1296, relu_53, primals_224, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1296 = primals_224 = None
    getitem_1299: "f32[8, 216, 42, 42]" = convolution_backward_271[0]
    getitem_1300: "f32[216, 1, 3, 3]" = convolution_backward_271[1];  convolution_backward_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_146: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_146: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_146, full_default, getitem_1299);  le_146 = getitem_1299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1167: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_57, where_146);  slice_57 = where_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_294: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1167, [0, 2, 3])
    sub_785: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_2558);  convolution_100 = unsqueeze_2558 = None
    mul_2721: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1167, sub_785)
    sum_295: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2721, [0, 2, 3]);  mul_2721 = None
    mul_2722: "f32[216]" = torch.ops.aten.mul.Tensor(sum_294, 7.086167800453515e-05)
    unsqueeze_2559: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2722, 0);  mul_2722 = None
    unsqueeze_2560: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2559, 2);  unsqueeze_2559 = None
    unsqueeze_2561: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2560, 3);  unsqueeze_2560 = None
    mul_2723: "f32[216]" = torch.ops.aten.mul.Tensor(sum_295, 7.086167800453515e-05)
    mul_2724: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_2725: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2723, mul_2724);  mul_2723 = mul_2724 = None
    unsqueeze_2562: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2725, 0);  mul_2725 = None
    unsqueeze_2563: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2562, 2);  unsqueeze_2562 = None
    unsqueeze_2564: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2563, 3);  unsqueeze_2563 = None
    mul_2726: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_222);  primals_222 = None
    unsqueeze_2565: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2726, 0);  mul_2726 = None
    unsqueeze_2566: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2565, 2);  unsqueeze_2565 = None
    unsqueeze_2567: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2566, 3);  unsqueeze_2566 = None
    mul_2727: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_785, unsqueeze_2564);  sub_785 = unsqueeze_2564 = None
    sub_787: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1167, mul_2727);  mul_2727 = None
    sub_788: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_787, unsqueeze_2561);  sub_787 = None
    mul_2728: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_788, unsqueeze_2567);  sub_788 = unsqueeze_2567 = None
    mul_2729: "f32[216]" = torch.ops.aten.mul.Tensor(sum_295, squeeze_163);  sum_295 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_272 = torch.ops.aten.convolution_backward.default(mul_2728, convolution_99, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2728 = convolution_99 = primals_221 = None
    getitem_1302: "f32[8, 216, 42, 42]" = convolution_backward_272[0]
    getitem_1303: "f32[216, 216, 1, 1]" = convolution_backward_272[1];  convolution_backward_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_273 = torch.ops.aten.convolution_backward.default(getitem_1302, relu_52, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1302 = primals_220 = None
    getitem_1305: "f32[8, 216, 42, 42]" = convolution_backward_273[0]
    getitem_1306: "f32[216, 1, 3, 3]" = convolution_backward_273[1];  convolution_backward_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_147: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_147: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_147, full_default, getitem_1305);  le_147 = getitem_1305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_296: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_147, [0, 2, 3])
    sub_789: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_2570);  convolution_98 = unsqueeze_2570 = None
    mul_2730: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_147, sub_789)
    sum_297: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2730, [0, 2, 3]);  mul_2730 = None
    mul_2731: "f32[216]" = torch.ops.aten.mul.Tensor(sum_296, 7.086167800453515e-05)
    unsqueeze_2571: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2731, 0);  mul_2731 = None
    unsqueeze_2572: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2571, 2);  unsqueeze_2571 = None
    unsqueeze_2573: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2572, 3);  unsqueeze_2572 = None
    mul_2732: "f32[216]" = torch.ops.aten.mul.Tensor(sum_297, 7.086167800453515e-05)
    mul_2733: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_2734: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2732, mul_2733);  mul_2732 = mul_2733 = None
    unsqueeze_2574: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2734, 0);  mul_2734 = None
    unsqueeze_2575: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2574, 2);  unsqueeze_2574 = None
    unsqueeze_2576: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2575, 3);  unsqueeze_2575 = None
    mul_2735: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_218);  primals_218 = None
    unsqueeze_2577: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2735, 0);  mul_2735 = None
    unsqueeze_2578: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2577, 2);  unsqueeze_2577 = None
    unsqueeze_2579: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2578, 3);  unsqueeze_2578 = None
    mul_2736: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_789, unsqueeze_2576);  sub_789 = unsqueeze_2576 = None
    sub_791: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_147, mul_2736);  where_147 = mul_2736 = None
    sub_792: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_791, unsqueeze_2573);  sub_791 = unsqueeze_2573 = None
    mul_2737: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_792, unsqueeze_2579);  sub_792 = unsqueeze_2579 = None
    mul_2738: "f32[216]" = torch.ops.aten.mul.Tensor(sum_297, squeeze_160);  sum_297 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_274 = torch.ops.aten.convolution_backward.default(mul_2737, convolution_97, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2737 = convolution_97 = primals_217 = None
    getitem_1308: "f32[8, 216, 42, 42]" = convolution_backward_274[0]
    getitem_1309: "f32[216, 216, 1, 1]" = convolution_backward_274[1];  convolution_backward_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_275 = torch.ops.aten.convolution_backward.default(getitem_1308, relu_47, primals_216, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1308 = primals_216 = None
    getitem_1311: "f32[8, 216, 42, 42]" = convolution_backward_275[0]
    getitem_1312: "f32[216, 1, 3, 3]" = convolution_backward_275[1];  convolution_backward_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_148: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_47, 0)
    where_148: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_148, full_default, getitem_1311);  getitem_1311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1168: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1166, where_148);  add_1166 = where_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_793: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_2582);  convolution_96 = unsqueeze_2582 = None
    mul_2739: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1167, sub_793)
    sum_299: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2739, [0, 2, 3]);  mul_2739 = None
    mul_2741: "f32[216]" = torch.ops.aten.mul.Tensor(sum_299, 7.086167800453515e-05)
    mul_2742: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_2743: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2741, mul_2742);  mul_2741 = mul_2742 = None
    unsqueeze_2586: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2743, 0);  mul_2743 = None
    unsqueeze_2587: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2586, 2);  unsqueeze_2586 = None
    unsqueeze_2588: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2587, 3);  unsqueeze_2587 = None
    mul_2744: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_214);  primals_214 = None
    unsqueeze_2589: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2744, 0);  mul_2744 = None
    unsqueeze_2590: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2589, 2);  unsqueeze_2589 = None
    unsqueeze_2591: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2590, 3);  unsqueeze_2590 = None
    mul_2745: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_793, unsqueeze_2588);  sub_793 = unsqueeze_2588 = None
    sub_795: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1167, mul_2745);  add_1167 = mul_2745 = None
    sub_796: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_795, unsqueeze_2561);  sub_795 = unsqueeze_2561 = None
    mul_2746: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_796, unsqueeze_2591);  sub_796 = unsqueeze_2591 = None
    mul_2747: "f32[216]" = torch.ops.aten.mul.Tensor(sum_299, squeeze_157);  sum_299 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_276 = torch.ops.aten.convolution_backward.default(mul_2746, convolution_95, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2746 = convolution_95 = primals_213 = None
    getitem_1314: "f32[8, 216, 42, 42]" = convolution_backward_276[0]
    getitem_1315: "f32[216, 216, 1, 1]" = convolution_backward_276[1];  convolution_backward_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_277 = torch.ops.aten.convolution_backward.default(getitem_1314, relu_50, primals_212, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1314 = primals_212 = None
    getitem_1317: "f32[8, 216, 42, 42]" = convolution_backward_277[0]
    getitem_1318: "f32[216, 1, 5, 5]" = convolution_backward_277[1];  convolution_backward_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_149: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_149: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_149, full_default, getitem_1317);  le_149 = getitem_1317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_300: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_149, [0, 2, 3])
    sub_797: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_2594);  convolution_94 = unsqueeze_2594 = None
    mul_2748: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_149, sub_797)
    sum_301: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2748, [0, 2, 3]);  mul_2748 = None
    mul_2749: "f32[216]" = torch.ops.aten.mul.Tensor(sum_300, 7.086167800453515e-05)
    unsqueeze_2595: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2749, 0);  mul_2749 = None
    unsqueeze_2596: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2595, 2);  unsqueeze_2595 = None
    unsqueeze_2597: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2596, 3);  unsqueeze_2596 = None
    mul_2750: "f32[216]" = torch.ops.aten.mul.Tensor(sum_301, 7.086167800453515e-05)
    mul_2751: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_2752: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2750, mul_2751);  mul_2750 = mul_2751 = None
    unsqueeze_2598: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2752, 0);  mul_2752 = None
    unsqueeze_2599: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2598, 2);  unsqueeze_2598 = None
    unsqueeze_2600: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2599, 3);  unsqueeze_2599 = None
    mul_2753: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_210);  primals_210 = None
    unsqueeze_2601: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2753, 0);  mul_2753 = None
    unsqueeze_2602: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2601, 2);  unsqueeze_2601 = None
    unsqueeze_2603: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2602, 3);  unsqueeze_2602 = None
    mul_2754: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_797, unsqueeze_2600);  sub_797 = unsqueeze_2600 = None
    sub_799: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_149, mul_2754);  where_149 = mul_2754 = None
    sub_800: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_799, unsqueeze_2597);  sub_799 = unsqueeze_2597 = None
    mul_2755: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_800, unsqueeze_2603);  sub_800 = unsqueeze_2603 = None
    mul_2756: "f32[216]" = torch.ops.aten.mul.Tensor(sum_301, squeeze_154);  sum_301 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_278 = torch.ops.aten.convolution_backward.default(mul_2755, convolution_93, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2755 = convolution_93 = primals_209 = None
    getitem_1320: "f32[8, 216, 42, 42]" = convolution_backward_278[0]
    getitem_1321: "f32[216, 216, 1, 1]" = convolution_backward_278[1];  convolution_backward_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_279 = torch.ops.aten.convolution_backward.default(getitem_1320, relu_47, primals_208, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1320 = primals_208 = None
    getitem_1323: "f32[8, 216, 42, 42]" = convolution_backward_279[0]
    getitem_1324: "f32[216, 1, 5, 5]" = convolution_backward_279[1];  convolution_backward_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_150: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_148, full_default, getitem_1323);  getitem_1323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1169: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1168, where_150);  add_1168 = where_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_31: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_56, add_249, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_123);  add_249 = getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1170: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1169, max_pool2d_with_indices_backward_31);  add_1169 = max_pool2d_with_indices_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_302: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_56, [0, 2, 3])
    sub_801: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_2606);  convolution_92 = unsqueeze_2606 = None
    mul_2757: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_56, sub_801)
    sum_303: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2757, [0, 2, 3]);  mul_2757 = None
    mul_2758: "f32[216]" = torch.ops.aten.mul.Tensor(sum_302, 7.086167800453515e-05)
    unsqueeze_2607: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2758, 0);  mul_2758 = None
    unsqueeze_2608: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2607, 2);  unsqueeze_2607 = None
    unsqueeze_2609: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2608, 3);  unsqueeze_2608 = None
    mul_2759: "f32[216]" = torch.ops.aten.mul.Tensor(sum_303, 7.086167800453515e-05)
    mul_2760: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_2761: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2759, mul_2760);  mul_2759 = mul_2760 = None
    unsqueeze_2610: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2761, 0);  mul_2761 = None
    unsqueeze_2611: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2610, 2);  unsqueeze_2610 = None
    unsqueeze_2612: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2611, 3);  unsqueeze_2611 = None
    mul_2762: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_206);  primals_206 = None
    unsqueeze_2613: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2762, 0);  mul_2762 = None
    unsqueeze_2614: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2613, 2);  unsqueeze_2613 = None
    unsqueeze_2615: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2614, 3);  unsqueeze_2614 = None
    mul_2763: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_801, unsqueeze_2612);  sub_801 = unsqueeze_2612 = None
    sub_803: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_56, mul_2763);  slice_56 = mul_2763 = None
    sub_804: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_803, unsqueeze_2609);  sub_803 = unsqueeze_2609 = None
    mul_2764: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_804, unsqueeze_2615);  sub_804 = unsqueeze_2615 = None
    mul_2765: "f32[216]" = torch.ops.aten.mul.Tensor(sum_303, squeeze_151);  sum_303 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_280 = torch.ops.aten.convolution_backward.default(mul_2764, convolution_91, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2764 = convolution_91 = primals_205 = None
    getitem_1326: "f32[8, 216, 42, 42]" = convolution_backward_280[0]
    getitem_1327: "f32[216, 216, 1, 1]" = convolution_backward_280[1];  convolution_backward_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_281 = torch.ops.aten.convolution_backward.default(getitem_1326, relu_48, primals_204, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1326 = primals_204 = None
    getitem_1329: "f32[8, 216, 42, 42]" = convolution_backward_281[0]
    getitem_1330: "f32[216, 1, 7, 7]" = convolution_backward_281[1];  convolution_backward_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_151: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_151: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_151, full_default, getitem_1329);  le_151 = getitem_1329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_304: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_151, [0, 2, 3])
    sub_805: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_2618);  convolution_90 = unsqueeze_2618 = None
    mul_2766: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_151, sub_805)
    sum_305: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2766, [0, 2, 3]);  mul_2766 = None
    mul_2767: "f32[216]" = torch.ops.aten.mul.Tensor(sum_304, 7.086167800453515e-05)
    unsqueeze_2619: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2767, 0);  mul_2767 = None
    unsqueeze_2620: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2619, 2);  unsqueeze_2619 = None
    unsqueeze_2621: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2620, 3);  unsqueeze_2620 = None
    mul_2768: "f32[216]" = torch.ops.aten.mul.Tensor(sum_305, 7.086167800453515e-05)
    mul_2769: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_2770: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2768, mul_2769);  mul_2768 = mul_2769 = None
    unsqueeze_2622: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2770, 0);  mul_2770 = None
    unsqueeze_2623: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2622, 2);  unsqueeze_2622 = None
    unsqueeze_2624: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2623, 3);  unsqueeze_2623 = None
    mul_2771: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_202);  primals_202 = None
    unsqueeze_2625: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2771, 0);  mul_2771 = None
    unsqueeze_2626: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2625, 2);  unsqueeze_2625 = None
    unsqueeze_2627: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2626, 3);  unsqueeze_2626 = None
    mul_2772: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_805, unsqueeze_2624);  sub_805 = unsqueeze_2624 = None
    sub_807: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_151, mul_2772);  where_151 = mul_2772 = None
    sub_808: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_807, unsqueeze_2621);  sub_807 = unsqueeze_2621 = None
    mul_2773: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_808, unsqueeze_2627);  sub_808 = unsqueeze_2627 = None
    mul_2774: "f32[216]" = torch.ops.aten.mul.Tensor(sum_305, squeeze_148);  sum_305 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_282 = torch.ops.aten.convolution_backward.default(mul_2773, convolution_89, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2773 = convolution_89 = primals_201 = None
    getitem_1332: "f32[8, 216, 42, 42]" = convolution_backward_282[0]
    getitem_1333: "f32[216, 216, 1, 1]" = convolution_backward_282[1];  convolution_backward_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_283 = torch.ops.aten.convolution_backward.default(getitem_1332, relu_47, primals_200, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1332 = relu_47 = primals_200 = None
    getitem_1335: "f32[8, 216, 42, 42]" = convolution_backward_283[0]
    getitem_1336: "f32[216, 1, 7, 7]" = convolution_backward_283[1];  convolution_backward_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_152: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_148, full_default, getitem_1335);  le_148 = getitem_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1171: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1170, where_152);  add_1170 = where_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_32: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_55, add_244, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_117);  add_244 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1172: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(where_144, max_pool2d_with_indices_backward_32);  where_144 = max_pool2d_with_indices_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_306: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_55, [0, 2, 3])
    sub_809: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_2630);  convolution_88 = unsqueeze_2630 = None
    mul_2775: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_55, sub_809)
    sum_307: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2775, [0, 2, 3]);  mul_2775 = None
    mul_2776: "f32[216]" = torch.ops.aten.mul.Tensor(sum_306, 7.086167800453515e-05)
    unsqueeze_2631: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2776, 0);  mul_2776 = None
    unsqueeze_2632: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2631, 2);  unsqueeze_2631 = None
    unsqueeze_2633: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2632, 3);  unsqueeze_2632 = None
    mul_2777: "f32[216]" = torch.ops.aten.mul.Tensor(sum_307, 7.086167800453515e-05)
    mul_2778: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_2779: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2777, mul_2778);  mul_2777 = mul_2778 = None
    unsqueeze_2634: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2779, 0);  mul_2779 = None
    unsqueeze_2635: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2634, 2);  unsqueeze_2634 = None
    unsqueeze_2636: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2635, 3);  unsqueeze_2635 = None
    mul_2780: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_198);  primals_198 = None
    unsqueeze_2637: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2780, 0);  mul_2780 = None
    unsqueeze_2638: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2637, 2);  unsqueeze_2637 = None
    unsqueeze_2639: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2638, 3);  unsqueeze_2638 = None
    mul_2781: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_809, unsqueeze_2636);  sub_809 = unsqueeze_2636 = None
    sub_811: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_55, mul_2781);  slice_55 = mul_2781 = None
    sub_812: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_811, unsqueeze_2633);  sub_811 = unsqueeze_2633 = None
    mul_2782: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_812, unsqueeze_2639);  sub_812 = unsqueeze_2639 = None
    mul_2783: "f32[216]" = torch.ops.aten.mul.Tensor(sum_307, squeeze_145);  sum_307 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_284 = torch.ops.aten.convolution_backward.default(mul_2782, convolution_87, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2782 = convolution_87 = primals_197 = None
    getitem_1338: "f32[8, 216, 42, 42]" = convolution_backward_284[0]
    getitem_1339: "f32[216, 216, 1, 1]" = convolution_backward_284[1];  convolution_backward_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_285 = torch.ops.aten.convolution_backward.default(getitem_1338, relu_46, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1338 = primals_196 = None
    getitem_1341: "f32[8, 216, 42, 42]" = convolution_backward_285[0]
    getitem_1342: "f32[216, 1, 5, 5]" = convolution_backward_285[1];  convolution_backward_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_153: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_153: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_153, full_default, getitem_1341);  le_153 = getitem_1341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_308: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_153, [0, 2, 3])
    sub_813: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_2642);  convolution_86 = unsqueeze_2642 = None
    mul_2784: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_153, sub_813)
    sum_309: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2784, [0, 2, 3]);  mul_2784 = None
    mul_2785: "f32[216]" = torch.ops.aten.mul.Tensor(sum_308, 7.086167800453515e-05)
    unsqueeze_2643: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2785, 0);  mul_2785 = None
    unsqueeze_2644: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2643, 2);  unsqueeze_2643 = None
    unsqueeze_2645: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2644, 3);  unsqueeze_2644 = None
    mul_2786: "f32[216]" = torch.ops.aten.mul.Tensor(sum_309, 7.086167800453515e-05)
    mul_2787: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_2788: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2786, mul_2787);  mul_2786 = mul_2787 = None
    unsqueeze_2646: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2788, 0);  mul_2788 = None
    unsqueeze_2647: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2646, 2);  unsqueeze_2646 = None
    unsqueeze_2648: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2647, 3);  unsqueeze_2647 = None
    mul_2789: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_194);  primals_194 = None
    unsqueeze_2649: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2789, 0);  mul_2789 = None
    unsqueeze_2650: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2649, 2);  unsqueeze_2649 = None
    unsqueeze_2651: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2650, 3);  unsqueeze_2650 = None
    mul_2790: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_813, unsqueeze_2648);  sub_813 = unsqueeze_2648 = None
    sub_815: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_153, mul_2790);  where_153 = mul_2790 = None
    sub_816: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_815, unsqueeze_2645);  sub_815 = unsqueeze_2645 = None
    mul_2791: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_816, unsqueeze_2651);  sub_816 = unsqueeze_2651 = None
    mul_2792: "f32[216]" = torch.ops.aten.mul.Tensor(sum_309, squeeze_142);  sum_309 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_286 = torch.ops.aten.convolution_backward.default(mul_2791, convolution_85, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2791 = convolution_85 = primals_193 = None
    getitem_1344: "f32[8, 216, 42, 42]" = convolution_backward_286[0]
    getitem_1345: "f32[216, 216, 1, 1]" = convolution_backward_286[1];  convolution_backward_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_287 = torch.ops.aten.convolution_backward.default(getitem_1344, relu_45, primals_192, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1344 = relu_45 = primals_192 = None
    getitem_1347: "f32[8, 216, 42, 42]" = convolution_backward_287[0]
    getitem_1348: "f32[216, 1, 5, 5]" = convolution_backward_287[1];  convolution_backward_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_154: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_144, full_default, getitem_1347);  le_144 = getitem_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1173: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1172, where_154);  add_1172 = where_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_310: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1171, [0, 2, 3])
    sub_817: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_2654);  convolution_84 = unsqueeze_2654 = None
    mul_2793: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1171, sub_817)
    sum_311: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2793, [0, 2, 3]);  mul_2793 = None
    mul_2794: "f32[216]" = torch.ops.aten.mul.Tensor(sum_310, 7.086167800453515e-05)
    unsqueeze_2655: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2794, 0);  mul_2794 = None
    unsqueeze_2656: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2655, 2);  unsqueeze_2655 = None
    unsqueeze_2657: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2656, 3);  unsqueeze_2656 = None
    mul_2795: "f32[216]" = torch.ops.aten.mul.Tensor(sum_311, 7.086167800453515e-05)
    mul_2796: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_2797: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2795, mul_2796);  mul_2795 = mul_2796 = None
    unsqueeze_2658: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2797, 0);  mul_2797 = None
    unsqueeze_2659: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2658, 2);  unsqueeze_2658 = None
    unsqueeze_2660: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2659, 3);  unsqueeze_2659 = None
    mul_2798: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_190);  primals_190 = None
    unsqueeze_2661: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2798, 0);  mul_2798 = None
    unsqueeze_2662: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2661, 2);  unsqueeze_2661 = None
    unsqueeze_2663: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2662, 3);  unsqueeze_2662 = None
    mul_2799: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_817, unsqueeze_2660);  sub_817 = unsqueeze_2660 = None
    sub_819: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1171, mul_2799);  add_1171 = mul_2799 = None
    sub_820: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_819, unsqueeze_2657);  sub_819 = unsqueeze_2657 = None
    mul_2800: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_820, unsqueeze_2663);  sub_820 = unsqueeze_2663 = None
    mul_2801: "f32[216]" = torch.ops.aten.mul.Tensor(sum_311, squeeze_139);  sum_311 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_288 = torch.ops.aten.convolution_backward.default(mul_2800, relu_44, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2800 = relu_44 = primals_189 = None
    getitem_1350: "f32[8, 1080, 42, 42]" = convolution_backward_288[0]
    getitem_1351: "f32[216, 1080, 1, 1]" = convolution_backward_288[1];  convolution_backward_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_155: "f32[8, 1080, 42, 42]" = torch.ops.aten.where.self(le_142, full_default, getitem_1350);  le_142 = getitem_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1174: "f32[8, 1080, 42, 42]" = torch.ops.aten.add.Tensor(where_142, where_155);  where_142 = where_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_312: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1173, [0, 2, 3])
    sub_821: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_2666);  convolution_83 = unsqueeze_2666 = None
    mul_2802: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1173, sub_821)
    sum_313: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2802, [0, 2, 3]);  mul_2802 = None
    mul_2803: "f32[216]" = torch.ops.aten.mul.Tensor(sum_312, 7.086167800453515e-05)
    unsqueeze_2667: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2803, 0);  mul_2803 = None
    unsqueeze_2668: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2667, 2);  unsqueeze_2667 = None
    unsqueeze_2669: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2668, 3);  unsqueeze_2668 = None
    mul_2804: "f32[216]" = torch.ops.aten.mul.Tensor(sum_313, 7.086167800453515e-05)
    mul_2805: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_2806: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2804, mul_2805);  mul_2804 = mul_2805 = None
    unsqueeze_2670: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2806, 0);  mul_2806 = None
    unsqueeze_2671: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2670, 2);  unsqueeze_2670 = None
    unsqueeze_2672: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2671, 3);  unsqueeze_2671 = None
    mul_2807: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_187);  primals_187 = None
    unsqueeze_2673: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2807, 0);  mul_2807 = None
    unsqueeze_2674: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2673, 2);  unsqueeze_2673 = None
    unsqueeze_2675: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2674, 3);  unsqueeze_2674 = None
    mul_2808: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_821, unsqueeze_2672);  sub_821 = unsqueeze_2672 = None
    sub_823: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1173, mul_2808);  add_1173 = mul_2808 = None
    sub_824: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_823, unsqueeze_2669);  sub_823 = unsqueeze_2669 = None
    mul_2809: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_824, unsqueeze_2675);  sub_824 = unsqueeze_2675 = None
    mul_2810: "f32[216]" = torch.ops.aten.mul.Tensor(sum_313, squeeze_136);  sum_313 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_289 = torch.ops.aten.convolution_backward.default(mul_2809, relu_30, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2809 = primals_186 = None
    getitem_1353: "f32[8, 540, 42, 42]" = convolution_backward_289[0]
    getitem_1354: "f32[216, 540, 1, 1]" = convolution_backward_289[1];  convolution_backward_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    le_156: "b8[8, 540, 42, 42]" = torch.ops.aten.le.Scalar(relu_30, 0)
    where_156: "f32[8, 540, 42, 42]" = torch.ops.aten.where.self(le_156, full_default, getitem_1353);  getitem_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_60: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1174, 1, 0, 216)
    slice_61: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1174, 1, 216, 432)
    slice_62: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1174, 1, 432, 648)
    slice_63: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1174, 1, 648, 864)
    slice_64: "f32[8, 216, 42, 42]" = torch.ops.aten.slice.Tensor(add_1174, 1, 864, 1080);  add_1174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_314: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_64, [0, 2, 3])
    sub_825: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_2678);  convolution_82 = unsqueeze_2678 = None
    mul_2811: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_64, sub_825)
    sum_315: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2811, [0, 2, 3]);  mul_2811 = None
    mul_2812: "f32[216]" = torch.ops.aten.mul.Tensor(sum_314, 7.086167800453515e-05)
    unsqueeze_2679: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2812, 0);  mul_2812 = None
    unsqueeze_2680: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2679, 2);  unsqueeze_2679 = None
    unsqueeze_2681: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2680, 3);  unsqueeze_2680 = None
    mul_2813: "f32[216]" = torch.ops.aten.mul.Tensor(sum_315, 7.086167800453515e-05)
    mul_2814: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_2815: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2813, mul_2814);  mul_2813 = mul_2814 = None
    unsqueeze_2682: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2815, 0);  mul_2815 = None
    unsqueeze_2683: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2682, 2);  unsqueeze_2682 = None
    unsqueeze_2684: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2683, 3);  unsqueeze_2683 = None
    mul_2816: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_184);  primals_184 = None
    unsqueeze_2685: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2816, 0);  mul_2816 = None
    unsqueeze_2686: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2685, 2);  unsqueeze_2685 = None
    unsqueeze_2687: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2686, 3);  unsqueeze_2686 = None
    mul_2817: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_825, unsqueeze_2684);  sub_825 = unsqueeze_2684 = None
    sub_827: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_64, mul_2817);  mul_2817 = None
    sub_828: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_827, unsqueeze_2681);  sub_827 = unsqueeze_2681 = None
    mul_2818: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_828, unsqueeze_2687);  sub_828 = unsqueeze_2687 = None
    mul_2819: "f32[216]" = torch.ops.aten.mul.Tensor(sum_315, squeeze_133);  sum_315 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_290 = torch.ops.aten.convolution_backward.default(mul_2818, convolution_81, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2818 = convolution_81 = primals_183 = None
    getitem_1356: "f32[8, 216, 42, 42]" = convolution_backward_290[0]
    getitem_1357: "f32[216, 216, 1, 1]" = convolution_backward_290[1];  convolution_backward_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_291 = torch.ops.aten.convolution_backward.default(getitem_1356, relu_42, primals_182, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1356 = primals_182 = None
    getitem_1359: "f32[8, 216, 42, 42]" = convolution_backward_291[0]
    getitem_1360: "f32[216, 1, 3, 3]" = convolution_backward_291[1];  convolution_backward_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_157: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_157: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_157, full_default, getitem_1359);  le_157 = getitem_1359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_316: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_157, [0, 2, 3])
    sub_829: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_2690);  convolution_80 = unsqueeze_2690 = None
    mul_2820: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_157, sub_829)
    sum_317: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2820, [0, 2, 3]);  mul_2820 = None
    mul_2821: "f32[216]" = torch.ops.aten.mul.Tensor(sum_316, 7.086167800453515e-05)
    unsqueeze_2691: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2821, 0);  mul_2821 = None
    unsqueeze_2692: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2691, 2);  unsqueeze_2691 = None
    unsqueeze_2693: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2692, 3);  unsqueeze_2692 = None
    mul_2822: "f32[216]" = torch.ops.aten.mul.Tensor(sum_317, 7.086167800453515e-05)
    mul_2823: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_2824: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2822, mul_2823);  mul_2822 = mul_2823 = None
    unsqueeze_2694: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2824, 0);  mul_2824 = None
    unsqueeze_2695: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2694, 2);  unsqueeze_2694 = None
    unsqueeze_2696: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2695, 3);  unsqueeze_2695 = None
    mul_2825: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_180);  primals_180 = None
    unsqueeze_2697: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2825, 0);  mul_2825 = None
    unsqueeze_2698: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2697, 2);  unsqueeze_2697 = None
    unsqueeze_2699: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2698, 3);  unsqueeze_2698 = None
    mul_2826: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_829, unsqueeze_2696);  sub_829 = unsqueeze_2696 = None
    sub_831: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_157, mul_2826);  where_157 = mul_2826 = None
    sub_832: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_831, unsqueeze_2693);  sub_831 = unsqueeze_2693 = None
    mul_2827: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_832, unsqueeze_2699);  sub_832 = unsqueeze_2699 = None
    mul_2828: "f32[216]" = torch.ops.aten.mul.Tensor(sum_317, squeeze_130);  sum_317 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_292 = torch.ops.aten.convolution_backward.default(mul_2827, convolution_79, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2827 = convolution_79 = primals_179 = None
    getitem_1362: "f32[8, 216, 42, 42]" = convolution_backward_292[0]
    getitem_1363: "f32[216, 216, 1, 1]" = convolution_backward_292[1];  convolution_backward_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_293 = torch.ops.aten.convolution_backward.default(getitem_1362, relu_31, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1362 = primals_178 = None
    getitem_1365: "f32[8, 216, 42, 42]" = convolution_backward_293[0]
    getitem_1366: "f32[216, 1, 3, 3]" = convolution_backward_293[1];  convolution_backward_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_158: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_31, 0)
    where_158: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_158, full_default, getitem_1365);  getitem_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    max_pool2d_with_indices_backward_33: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_63, add_174, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:118, code: x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    add_1175: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_64, max_pool2d_with_indices_backward_33);  slice_64 = max_pool2d_with_indices_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_318: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_63, [0, 2, 3])
    sub_833: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_2702);  convolution_78 = unsqueeze_2702 = None
    mul_2829: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_63, sub_833)
    sum_319: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2829, [0, 2, 3]);  mul_2829 = None
    mul_2830: "f32[216]" = torch.ops.aten.mul.Tensor(sum_318, 7.086167800453515e-05)
    unsqueeze_2703: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2830, 0);  mul_2830 = None
    unsqueeze_2704: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2703, 2);  unsqueeze_2703 = None
    unsqueeze_2705: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2704, 3);  unsqueeze_2704 = None
    mul_2831: "f32[216]" = torch.ops.aten.mul.Tensor(sum_319, 7.086167800453515e-05)
    mul_2832: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_2833: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2831, mul_2832);  mul_2831 = mul_2832 = None
    unsqueeze_2706: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2833, 0);  mul_2833 = None
    unsqueeze_2707: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2706, 2);  unsqueeze_2706 = None
    unsqueeze_2708: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2707, 3);  unsqueeze_2707 = None
    mul_2834: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_176);  primals_176 = None
    unsqueeze_2709: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2834, 0);  mul_2834 = None
    unsqueeze_2710: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2709, 2);  unsqueeze_2709 = None
    unsqueeze_2711: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2710, 3);  unsqueeze_2710 = None
    mul_2835: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_833, unsqueeze_2708);  sub_833 = unsqueeze_2708 = None
    sub_835: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_63, mul_2835);  slice_63 = mul_2835 = None
    sub_836: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_835, unsqueeze_2705);  sub_835 = unsqueeze_2705 = None
    mul_2836: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_836, unsqueeze_2711);  sub_836 = unsqueeze_2711 = None
    mul_2837: "f32[216]" = torch.ops.aten.mul.Tensor(sum_319, squeeze_127);  sum_319 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_294 = torch.ops.aten.convolution_backward.default(mul_2836, convolution_77, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2836 = convolution_77 = primals_175 = None
    getitem_1368: "f32[8, 216, 42, 42]" = convolution_backward_294[0]
    getitem_1369: "f32[216, 216, 1, 1]" = convolution_backward_294[1];  convolution_backward_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_295 = torch.ops.aten.convolution_backward.default(getitem_1368, relu_40, primals_174, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1368 = primals_174 = None
    getitem_1371: "f32[8, 216, 42, 42]" = convolution_backward_295[0]
    getitem_1372: "f32[216, 1, 3, 3]" = convolution_backward_295[1];  convolution_backward_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_159: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_159: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_159, full_default, getitem_1371);  le_159 = getitem_1371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_320: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_159, [0, 2, 3])
    sub_837: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_2714);  convolution_76 = unsqueeze_2714 = None
    mul_2838: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_159, sub_837)
    sum_321: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2838, [0, 2, 3]);  mul_2838 = None
    mul_2839: "f32[216]" = torch.ops.aten.mul.Tensor(sum_320, 7.086167800453515e-05)
    unsqueeze_2715: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2839, 0);  mul_2839 = None
    unsqueeze_2716: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2715, 2);  unsqueeze_2715 = None
    unsqueeze_2717: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2716, 3);  unsqueeze_2716 = None
    mul_2840: "f32[216]" = torch.ops.aten.mul.Tensor(sum_321, 7.086167800453515e-05)
    mul_2841: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_2842: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2840, mul_2841);  mul_2840 = mul_2841 = None
    unsqueeze_2718: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2842, 0);  mul_2842 = None
    unsqueeze_2719: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2718, 2);  unsqueeze_2718 = None
    unsqueeze_2720: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2719, 3);  unsqueeze_2719 = None
    mul_2843: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_172);  primals_172 = None
    unsqueeze_2721: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2843, 0);  mul_2843 = None
    unsqueeze_2722: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2721, 2);  unsqueeze_2721 = None
    unsqueeze_2723: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2722, 3);  unsqueeze_2722 = None
    mul_2844: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_837, unsqueeze_2720);  sub_837 = unsqueeze_2720 = None
    sub_839: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_159, mul_2844);  where_159 = mul_2844 = None
    sub_840: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_839, unsqueeze_2717);  sub_839 = unsqueeze_2717 = None
    mul_2845: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_840, unsqueeze_2723);  sub_840 = unsqueeze_2723 = None
    mul_2846: "f32[216]" = torch.ops.aten.mul.Tensor(sum_321, squeeze_124);  sum_321 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_296 = torch.ops.aten.convolution_backward.default(mul_2845, convolution_75, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2845 = convolution_75 = primals_171 = None
    getitem_1374: "f32[8, 216, 42, 42]" = convolution_backward_296[0]
    getitem_1375: "f32[216, 216, 1, 1]" = convolution_backward_296[1];  convolution_backward_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_297 = torch.ops.aten.convolution_backward.default(getitem_1374, relu_39, primals_170, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1374 = primals_170 = None
    getitem_1377: "f32[8, 216, 42, 42]" = convolution_backward_297[0]
    getitem_1378: "f32[216, 1, 3, 3]" = convolution_backward_297[1];  convolution_backward_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_160: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_160: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_160, full_default, getitem_1377);  le_160 = getitem_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1176: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(slice_62, where_160);  slice_62 = where_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_322: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1176, [0, 2, 3])
    sub_841: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_2726);  convolution_74 = unsqueeze_2726 = None
    mul_2847: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1176, sub_841)
    sum_323: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2847, [0, 2, 3]);  mul_2847 = None
    mul_2848: "f32[216]" = torch.ops.aten.mul.Tensor(sum_322, 7.086167800453515e-05)
    unsqueeze_2727: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2848, 0);  mul_2848 = None
    unsqueeze_2728: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2727, 2);  unsqueeze_2727 = None
    unsqueeze_2729: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2728, 3);  unsqueeze_2728 = None
    mul_2849: "f32[216]" = torch.ops.aten.mul.Tensor(sum_323, 7.086167800453515e-05)
    mul_2850: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_2851: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2849, mul_2850);  mul_2849 = mul_2850 = None
    unsqueeze_2730: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2851, 0);  mul_2851 = None
    unsqueeze_2731: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2730, 2);  unsqueeze_2730 = None
    unsqueeze_2732: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2731, 3);  unsqueeze_2731 = None
    mul_2852: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_168);  primals_168 = None
    unsqueeze_2733: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2852, 0);  mul_2852 = None
    unsqueeze_2734: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2733, 2);  unsqueeze_2733 = None
    unsqueeze_2735: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2734, 3);  unsqueeze_2734 = None
    mul_2853: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_841, unsqueeze_2732);  sub_841 = unsqueeze_2732 = None
    sub_843: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1176, mul_2853);  mul_2853 = None
    sub_844: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_843, unsqueeze_2729);  sub_843 = None
    mul_2854: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_844, unsqueeze_2735);  sub_844 = unsqueeze_2735 = None
    mul_2855: "f32[216]" = torch.ops.aten.mul.Tensor(sum_323, squeeze_121);  sum_323 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_298 = torch.ops.aten.convolution_backward.default(mul_2854, convolution_73, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2854 = convolution_73 = primals_167 = None
    getitem_1380: "f32[8, 216, 42, 42]" = convolution_backward_298[0]
    getitem_1381: "f32[216, 216, 1, 1]" = convolution_backward_298[1];  convolution_backward_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_299 = torch.ops.aten.convolution_backward.default(getitem_1380, relu_38, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1380 = primals_166 = None
    getitem_1383: "f32[8, 216, 42, 42]" = convolution_backward_299[0]
    getitem_1384: "f32[216, 1, 3, 3]" = convolution_backward_299[1];  convolution_backward_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_161: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_161: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_161, full_default, getitem_1383);  le_161 = getitem_1383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_324: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_161, [0, 2, 3])
    sub_845: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_2738);  convolution_72 = unsqueeze_2738 = None
    mul_2856: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_161, sub_845)
    sum_325: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2856, [0, 2, 3]);  mul_2856 = None
    mul_2857: "f32[216]" = torch.ops.aten.mul.Tensor(sum_324, 7.086167800453515e-05)
    unsqueeze_2739: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2857, 0);  mul_2857 = None
    unsqueeze_2740: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2739, 2);  unsqueeze_2739 = None
    unsqueeze_2741: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2740, 3);  unsqueeze_2740 = None
    mul_2858: "f32[216]" = torch.ops.aten.mul.Tensor(sum_325, 7.086167800453515e-05)
    mul_2859: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_2860: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2858, mul_2859);  mul_2858 = mul_2859 = None
    unsqueeze_2742: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2860, 0);  mul_2860 = None
    unsqueeze_2743: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2742, 2);  unsqueeze_2742 = None
    unsqueeze_2744: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2743, 3);  unsqueeze_2743 = None
    mul_2861: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_164);  primals_164 = None
    unsqueeze_2745: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2861, 0);  mul_2861 = None
    unsqueeze_2746: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2745, 2);  unsqueeze_2745 = None
    unsqueeze_2747: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2746, 3);  unsqueeze_2746 = None
    mul_2862: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_845, unsqueeze_2744);  sub_845 = unsqueeze_2744 = None
    sub_847: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_161, mul_2862);  where_161 = mul_2862 = None
    sub_848: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_847, unsqueeze_2741);  sub_847 = unsqueeze_2741 = None
    mul_2863: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_848, unsqueeze_2747);  sub_848 = unsqueeze_2747 = None
    mul_2864: "f32[216]" = torch.ops.aten.mul.Tensor(sum_325, squeeze_118);  sum_325 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_300 = torch.ops.aten.convolution_backward.default(mul_2863, convolution_71, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2863 = convolution_71 = primals_163 = None
    getitem_1386: "f32[8, 216, 42, 42]" = convolution_backward_300[0]
    getitem_1387: "f32[216, 216, 1, 1]" = convolution_backward_300[1];  convolution_backward_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_301 = torch.ops.aten.convolution_backward.default(getitem_1386, relu_33, primals_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1386 = primals_162 = None
    getitem_1389: "f32[8, 216, 42, 42]" = convolution_backward_301[0]
    getitem_1390: "f32[216, 1, 3, 3]" = convolution_backward_301[1];  convolution_backward_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_162: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_33, 0)
    where_162: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_162, full_default, getitem_1389);  getitem_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1177: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1175, where_162);  add_1175 = where_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_849: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_2750);  convolution_70 = unsqueeze_2750 = None
    mul_2865: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1176, sub_849)
    sum_327: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2865, [0, 2, 3]);  mul_2865 = None
    mul_2867: "f32[216]" = torch.ops.aten.mul.Tensor(sum_327, 7.086167800453515e-05)
    mul_2868: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_2869: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2867, mul_2868);  mul_2867 = mul_2868 = None
    unsqueeze_2754: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2869, 0);  mul_2869 = None
    unsqueeze_2755: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2754, 2);  unsqueeze_2754 = None
    unsqueeze_2756: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2755, 3);  unsqueeze_2755 = None
    mul_2870: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_160);  primals_160 = None
    unsqueeze_2757: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2870, 0);  mul_2870 = None
    unsqueeze_2758: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2757, 2);  unsqueeze_2757 = None
    unsqueeze_2759: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2758, 3);  unsqueeze_2758 = None
    mul_2871: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_849, unsqueeze_2756);  sub_849 = unsqueeze_2756 = None
    sub_851: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1176, mul_2871);  add_1176 = mul_2871 = None
    sub_852: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_851, unsqueeze_2729);  sub_851 = unsqueeze_2729 = None
    mul_2872: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_852, unsqueeze_2759);  sub_852 = unsqueeze_2759 = None
    mul_2873: "f32[216]" = torch.ops.aten.mul.Tensor(sum_327, squeeze_115);  sum_327 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_302 = torch.ops.aten.convolution_backward.default(mul_2872, convolution_69, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2872 = convolution_69 = primals_159 = None
    getitem_1392: "f32[8, 216, 42, 42]" = convolution_backward_302[0]
    getitem_1393: "f32[216, 216, 1, 1]" = convolution_backward_302[1];  convolution_backward_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_303 = torch.ops.aten.convolution_backward.default(getitem_1392, relu_36, primals_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1392 = primals_158 = None
    getitem_1395: "f32[8, 216, 42, 42]" = convolution_backward_303[0]
    getitem_1396: "f32[216, 1, 5, 5]" = convolution_backward_303[1];  convolution_backward_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_163: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_163: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_163, full_default, getitem_1395);  le_163 = getitem_1395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_328: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_163, [0, 2, 3])
    sub_853: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_2762);  convolution_68 = unsqueeze_2762 = None
    mul_2874: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_163, sub_853)
    sum_329: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2874, [0, 2, 3]);  mul_2874 = None
    mul_2875: "f32[216]" = torch.ops.aten.mul.Tensor(sum_328, 7.086167800453515e-05)
    unsqueeze_2763: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2875, 0);  mul_2875 = None
    unsqueeze_2764: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2763, 2);  unsqueeze_2763 = None
    unsqueeze_2765: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2764, 3);  unsqueeze_2764 = None
    mul_2876: "f32[216]" = torch.ops.aten.mul.Tensor(sum_329, 7.086167800453515e-05)
    mul_2877: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_2878: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2876, mul_2877);  mul_2876 = mul_2877 = None
    unsqueeze_2766: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2878, 0);  mul_2878 = None
    unsqueeze_2767: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2766, 2);  unsqueeze_2766 = None
    unsqueeze_2768: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2767, 3);  unsqueeze_2767 = None
    mul_2879: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_156);  primals_156 = None
    unsqueeze_2769: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2879, 0);  mul_2879 = None
    unsqueeze_2770: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2769, 2);  unsqueeze_2769 = None
    unsqueeze_2771: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2770, 3);  unsqueeze_2770 = None
    mul_2880: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_853, unsqueeze_2768);  sub_853 = unsqueeze_2768 = None
    sub_855: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_163, mul_2880);  where_163 = mul_2880 = None
    sub_856: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_855, unsqueeze_2765);  sub_855 = unsqueeze_2765 = None
    mul_2881: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_856, unsqueeze_2771);  sub_856 = unsqueeze_2771 = None
    mul_2882: "f32[216]" = torch.ops.aten.mul.Tensor(sum_329, squeeze_112);  sum_329 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_304 = torch.ops.aten.convolution_backward.default(mul_2881, convolution_67, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2881 = convolution_67 = primals_155 = None
    getitem_1398: "f32[8, 216, 42, 42]" = convolution_backward_304[0]
    getitem_1399: "f32[216, 216, 1, 1]" = convolution_backward_304[1];  convolution_backward_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_305 = torch.ops.aten.convolution_backward.default(getitem_1398, relu_33, primals_154, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1398 = primals_154 = None
    getitem_1401: "f32[8, 216, 42, 42]" = convolution_backward_305[0]
    getitem_1402: "f32[216, 1, 5, 5]" = convolution_backward_305[1];  convolution_backward_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_164: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_162, full_default, getitem_1401);  getitem_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1178: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1177, where_164);  add_1177 = where_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    max_pool2d_with_indices_backward_34: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_61, add_174, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_89);  add_174 = getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:110, code: x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    add_1179: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1178, max_pool2d_with_indices_backward_34);  add_1178 = max_pool2d_with_indices_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_330: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_61, [0, 2, 3])
    sub_857: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_2774);  convolution_66 = unsqueeze_2774 = None
    mul_2883: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_61, sub_857)
    sum_331: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2883, [0, 2, 3]);  mul_2883 = None
    mul_2884: "f32[216]" = torch.ops.aten.mul.Tensor(sum_330, 7.086167800453515e-05)
    unsqueeze_2775: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2884, 0);  mul_2884 = None
    unsqueeze_2776: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2775, 2);  unsqueeze_2775 = None
    unsqueeze_2777: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2776, 3);  unsqueeze_2776 = None
    mul_2885: "f32[216]" = torch.ops.aten.mul.Tensor(sum_331, 7.086167800453515e-05)
    mul_2886: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_2887: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2885, mul_2886);  mul_2885 = mul_2886 = None
    unsqueeze_2778: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2887, 0);  mul_2887 = None
    unsqueeze_2779: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2778, 2);  unsqueeze_2778 = None
    unsqueeze_2780: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2779, 3);  unsqueeze_2779 = None
    mul_2888: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_152);  primals_152 = None
    unsqueeze_2781: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2888, 0);  mul_2888 = None
    unsqueeze_2782: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2781, 2);  unsqueeze_2781 = None
    unsqueeze_2783: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2782, 3);  unsqueeze_2782 = None
    mul_2889: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_857, unsqueeze_2780);  sub_857 = unsqueeze_2780 = None
    sub_859: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_61, mul_2889);  slice_61 = mul_2889 = None
    sub_860: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_859, unsqueeze_2777);  sub_859 = unsqueeze_2777 = None
    mul_2890: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_860, unsqueeze_2783);  sub_860 = unsqueeze_2783 = None
    mul_2891: "f32[216]" = torch.ops.aten.mul.Tensor(sum_331, squeeze_109);  sum_331 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_306 = torch.ops.aten.convolution_backward.default(mul_2890, convolution_65, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2890 = convolution_65 = primals_151 = None
    getitem_1404: "f32[8, 216, 42, 42]" = convolution_backward_306[0]
    getitem_1405: "f32[216, 216, 1, 1]" = convolution_backward_306[1];  convolution_backward_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_307 = torch.ops.aten.convolution_backward.default(getitem_1404, relu_34, primals_150, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1404 = primals_150 = None
    getitem_1407: "f32[8, 216, 42, 42]" = convolution_backward_307[0]
    getitem_1408: "f32[216, 1, 7, 7]" = convolution_backward_307[1];  convolution_backward_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_165: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_165: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_165, full_default, getitem_1407);  le_165 = getitem_1407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_332: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_165, [0, 2, 3])
    sub_861: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_2786);  convolution_64 = unsqueeze_2786 = None
    mul_2892: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_165, sub_861)
    sum_333: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2892, [0, 2, 3]);  mul_2892 = None
    mul_2893: "f32[216]" = torch.ops.aten.mul.Tensor(sum_332, 7.086167800453515e-05)
    unsqueeze_2787: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2893, 0);  mul_2893 = None
    unsqueeze_2788: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2787, 2);  unsqueeze_2787 = None
    unsqueeze_2789: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2788, 3);  unsqueeze_2788 = None
    mul_2894: "f32[216]" = torch.ops.aten.mul.Tensor(sum_333, 7.086167800453515e-05)
    mul_2895: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_2896: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2894, mul_2895);  mul_2894 = mul_2895 = None
    unsqueeze_2790: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2896, 0);  mul_2896 = None
    unsqueeze_2791: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2790, 2);  unsqueeze_2790 = None
    unsqueeze_2792: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2791, 3);  unsqueeze_2791 = None
    mul_2897: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_148);  primals_148 = None
    unsqueeze_2793: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2897, 0);  mul_2897 = None
    unsqueeze_2794: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2793, 2);  unsqueeze_2793 = None
    unsqueeze_2795: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2794, 3);  unsqueeze_2794 = None
    mul_2898: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_861, unsqueeze_2792);  sub_861 = unsqueeze_2792 = None
    sub_863: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_165, mul_2898);  where_165 = mul_2898 = None
    sub_864: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_863, unsqueeze_2789);  sub_863 = unsqueeze_2789 = None
    mul_2899: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_864, unsqueeze_2795);  sub_864 = unsqueeze_2795 = None
    mul_2900: "f32[216]" = torch.ops.aten.mul.Tensor(sum_333, squeeze_106);  sum_333 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_308 = torch.ops.aten.convolution_backward.default(mul_2899, convolution_63, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2899 = convolution_63 = primals_147 = None
    getitem_1410: "f32[8, 216, 42, 42]" = convolution_backward_308[0]
    getitem_1411: "f32[216, 216, 1, 1]" = convolution_backward_308[1];  convolution_backward_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_309 = torch.ops.aten.convolution_backward.default(getitem_1410, relu_33, primals_146, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1410 = relu_33 = primals_146 = None
    getitem_1413: "f32[8, 216, 42, 42]" = convolution_backward_309[0]
    getitem_1414: "f32[216, 1, 7, 7]" = convolution_backward_309[1];  convolution_backward_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_166: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_162, full_default, getitem_1413);  le_162 = getitem_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1180: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1179, where_166);  add_1179 = where_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    max_pool2d_with_indices_backward_35: "f32[8, 216, 42, 42]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_60, add_169, [3, 3], [1, 1], [1, 1], [1, 1], False, getitem_83);  add_169 = getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    add_1181: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(where_158, max_pool2d_with_indices_backward_35);  where_158 = max_pool2d_with_indices_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_334: "f32[216]" = torch.ops.aten.sum.dim_IntList(slice_60, [0, 2, 3])
    sub_865: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_2798);  convolution_62 = unsqueeze_2798 = None
    mul_2901: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(slice_60, sub_865)
    sum_335: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2901, [0, 2, 3]);  mul_2901 = None
    mul_2902: "f32[216]" = torch.ops.aten.mul.Tensor(sum_334, 7.086167800453515e-05)
    unsqueeze_2799: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2902, 0);  mul_2902 = None
    unsqueeze_2800: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2799, 2);  unsqueeze_2799 = None
    unsqueeze_2801: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2800, 3);  unsqueeze_2800 = None
    mul_2903: "f32[216]" = torch.ops.aten.mul.Tensor(sum_335, 7.086167800453515e-05)
    mul_2904: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_2905: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2903, mul_2904);  mul_2903 = mul_2904 = None
    unsqueeze_2802: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2905, 0);  mul_2905 = None
    unsqueeze_2803: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2802, 2);  unsqueeze_2802 = None
    unsqueeze_2804: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2803, 3);  unsqueeze_2803 = None
    mul_2906: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_144);  primals_144 = None
    unsqueeze_2805: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2906, 0);  mul_2906 = None
    unsqueeze_2806: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2805, 2);  unsqueeze_2805 = None
    unsqueeze_2807: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2806, 3);  unsqueeze_2806 = None
    mul_2907: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_865, unsqueeze_2804);  sub_865 = unsqueeze_2804 = None
    sub_867: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(slice_60, mul_2907);  slice_60 = mul_2907 = None
    sub_868: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_867, unsqueeze_2801);  sub_867 = unsqueeze_2801 = None
    mul_2908: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_868, unsqueeze_2807);  sub_868 = unsqueeze_2807 = None
    mul_2909: "f32[216]" = torch.ops.aten.mul.Tensor(sum_335, squeeze_103);  sum_335 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_310 = torch.ops.aten.convolution_backward.default(mul_2908, convolution_61, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2908 = convolution_61 = primals_143 = None
    getitem_1416: "f32[8, 216, 42, 42]" = convolution_backward_310[0]
    getitem_1417: "f32[216, 216, 1, 1]" = convolution_backward_310[1];  convolution_backward_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_311 = torch.ops.aten.convolution_backward.default(getitem_1416, relu_32, primals_142, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1416 = primals_142 = None
    getitem_1419: "f32[8, 216, 42, 42]" = convolution_backward_311[0]
    getitem_1420: "f32[216, 1, 5, 5]" = convolution_backward_311[1];  convolution_backward_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_167: "b8[8, 216, 42, 42]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_167: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_167, full_default, getitem_1419);  le_167 = getitem_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_336: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_167, [0, 2, 3])
    sub_869: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_2810);  convolution_60 = unsqueeze_2810 = None
    mul_2910: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(where_167, sub_869)
    sum_337: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2910, [0, 2, 3]);  mul_2910 = None
    mul_2911: "f32[216]" = torch.ops.aten.mul.Tensor(sum_336, 7.086167800453515e-05)
    unsqueeze_2811: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2911, 0);  mul_2911 = None
    unsqueeze_2812: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2811, 2);  unsqueeze_2811 = None
    unsqueeze_2813: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2812, 3);  unsqueeze_2812 = None
    mul_2912: "f32[216]" = torch.ops.aten.mul.Tensor(sum_337, 7.086167800453515e-05)
    mul_2913: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_2914: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2912, mul_2913);  mul_2912 = mul_2913 = None
    unsqueeze_2814: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2914, 0);  mul_2914 = None
    unsqueeze_2815: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2814, 2);  unsqueeze_2814 = None
    unsqueeze_2816: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2815, 3);  unsqueeze_2815 = None
    mul_2915: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_140);  primals_140 = None
    unsqueeze_2817: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2915, 0);  mul_2915 = None
    unsqueeze_2818: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2817, 2);  unsqueeze_2817 = None
    unsqueeze_2819: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2818, 3);  unsqueeze_2818 = None
    mul_2916: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_869, unsqueeze_2816);  sub_869 = unsqueeze_2816 = None
    sub_871: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(where_167, mul_2916);  where_167 = mul_2916 = None
    sub_872: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_871, unsqueeze_2813);  sub_871 = unsqueeze_2813 = None
    mul_2917: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_872, unsqueeze_2819);  sub_872 = unsqueeze_2819 = None
    mul_2918: "f32[216]" = torch.ops.aten.mul.Tensor(sum_337, squeeze_100);  sum_337 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_312 = torch.ops.aten.convolution_backward.default(mul_2917, convolution_59, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2917 = convolution_59 = primals_139 = None
    getitem_1422: "f32[8, 216, 42, 42]" = convolution_backward_312[0]
    getitem_1423: "f32[216, 216, 1, 1]" = convolution_backward_312[1];  convolution_backward_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_313 = torch.ops.aten.convolution_backward.default(getitem_1422, relu_31, primals_138, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 216, [True, True, False]);  getitem_1422 = relu_31 = primals_138 = None
    getitem_1425: "f32[8, 216, 42, 42]" = convolution_backward_313[0]
    getitem_1426: "f32[216, 1, 5, 5]" = convolution_backward_313[1];  convolution_backward_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_168: "f32[8, 216, 42, 42]" = torch.ops.aten.where.self(le_158, full_default, getitem_1425);  le_158 = getitem_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1182: "f32[8, 216, 42, 42]" = torch.ops.aten.add.Tensor(add_1181, where_168);  add_1181 = where_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_338: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1180, [0, 2, 3])
    sub_873: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_2822);  convolution_58 = unsqueeze_2822 = None
    mul_2919: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1180, sub_873)
    sum_339: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2919, [0, 2, 3]);  mul_2919 = None
    mul_2920: "f32[216]" = torch.ops.aten.mul.Tensor(sum_338, 7.086167800453515e-05)
    unsqueeze_2823: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2920, 0);  mul_2920 = None
    unsqueeze_2824: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2823, 2);  unsqueeze_2823 = None
    unsqueeze_2825: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2824, 3);  unsqueeze_2824 = None
    mul_2921: "f32[216]" = torch.ops.aten.mul.Tensor(sum_339, 7.086167800453515e-05)
    mul_2922: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_2923: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2921, mul_2922);  mul_2921 = mul_2922 = None
    unsqueeze_2826: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2923, 0);  mul_2923 = None
    unsqueeze_2827: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2826, 2);  unsqueeze_2826 = None
    unsqueeze_2828: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2827, 3);  unsqueeze_2827 = None
    mul_2924: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_136);  primals_136 = None
    unsqueeze_2829: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2924, 0);  mul_2924 = None
    unsqueeze_2830: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2829, 2);  unsqueeze_2829 = None
    unsqueeze_2831: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2830, 3);  unsqueeze_2830 = None
    mul_2925: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_873, unsqueeze_2828);  sub_873 = unsqueeze_2828 = None
    sub_875: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1180, mul_2925);  add_1180 = mul_2925 = None
    sub_876: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_875, unsqueeze_2825);  sub_875 = unsqueeze_2825 = None
    mul_2926: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_876, unsqueeze_2831);  sub_876 = unsqueeze_2831 = None
    mul_2927: "f32[216]" = torch.ops.aten.mul.Tensor(sum_339, squeeze_97);  sum_339 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_314 = torch.ops.aten.convolution_backward.default(mul_2926, relu_30, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2926 = relu_30 = primals_135 = None
    getitem_1428: "f32[8, 540, 42, 42]" = convolution_backward_314[0]
    getitem_1429: "f32[216, 540, 1, 1]" = convolution_backward_314[1];  convolution_backward_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_169: "f32[8, 540, 42, 42]" = torch.ops.aten.where.self(le_156, full_default, getitem_1428);  le_156 = getitem_1428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1183: "f32[8, 540, 42, 42]" = torch.ops.aten.add.Tensor(where_156, where_169);  where_156 = where_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    sum_340: "f32[216]" = torch.ops.aten.sum.dim_IntList(add_1182, [0, 2, 3])
    sub_877: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_2834);  cat_3 = unsqueeze_2834 = None
    mul_2928: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(add_1182, sub_877)
    sum_341: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_2928, [0, 2, 3]);  mul_2928 = None
    mul_2929: "f32[216]" = torch.ops.aten.mul.Tensor(sum_340, 7.086167800453515e-05)
    unsqueeze_2835: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2929, 0);  mul_2929 = None
    unsqueeze_2836: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2835, 2);  unsqueeze_2835 = None
    unsqueeze_2837: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2836, 3);  unsqueeze_2836 = None
    mul_2930: "f32[216]" = torch.ops.aten.mul.Tensor(sum_341, 7.086167800453515e-05)
    mul_2931: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_2932: "f32[216]" = torch.ops.aten.mul.Tensor(mul_2930, mul_2931);  mul_2930 = mul_2931 = None
    unsqueeze_2838: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2932, 0);  mul_2932 = None
    unsqueeze_2839: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2838, 2);  unsqueeze_2838 = None
    unsqueeze_2840: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2839, 3);  unsqueeze_2839 = None
    mul_2933: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_133);  primals_133 = None
    unsqueeze_2841: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_2933, 0);  mul_2933 = None
    unsqueeze_2842: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2841, 2);  unsqueeze_2841 = None
    unsqueeze_2843: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2842, 3);  unsqueeze_2842 = None
    mul_2934: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_877, unsqueeze_2840);  sub_877 = unsqueeze_2840 = None
    sub_879: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(add_1182, mul_2934);  add_1182 = mul_2934 = None
    sub_880: "f32[8, 216, 42, 42]" = torch.ops.aten.sub.Tensor(sub_879, unsqueeze_2837);  sub_879 = unsqueeze_2837 = None
    mul_2935: "f32[8, 216, 42, 42]" = torch.ops.aten.mul.Tensor(sub_880, unsqueeze_2843);  sub_880 = unsqueeze_2843 = None
    mul_2936: "f32[216]" = torch.ops.aten.mul.Tensor(sum_341, squeeze_94);  sum_341 = squeeze_94 = None
    slice_65: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(mul_2935, 1, 0, 108)
    slice_66: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(mul_2935, 1, 108, 216);  mul_2935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    convolution_backward_315 = torch.ops.aten.convolution_backward.default(slice_66, avg_pool2d_3, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_66 = avg_pool2d_3 = primals_132 = None
    getitem_1431: "f32[8, 270, 42, 42]" = convolution_backward_315[0]
    getitem_1432: "f32[108, 270, 1, 1]" = convolution_backward_315[1];  convolution_backward_315 = None
    avg_pool2d_backward_4: "f32[8, 270, 83, 83]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1431, constant_pad_nd_19, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1431 = constant_pad_nd_19 = None
    constant_pad_nd_60: "f32[8, 270, 83, 83]" = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_4, [1, -1, 1, -1]);  avg_pool2d_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    convolution_backward_316 = torch.ops.aten.convolution_backward.default(slice_65, avg_pool2d_2, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_65 = avg_pool2d_2 = primals_131 = None
    getitem_1434: "f32[8, 270, 42, 42]" = convolution_backward_316[0]
    getitem_1435: "f32[108, 270, 1, 1]" = convolution_backward_316[1];  convolution_backward_316 = None
    avg_pool2d_backward_5: "f32[8, 270, 83, 83]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1434, relu_15, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    add_1184: "f32[8, 270, 83, 83]" = torch.ops.aten.add.Tensor(constant_pad_nd_60, avg_pool2d_backward_5);  constant_pad_nd_60 = avg_pool2d_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    le_170: "b8[8, 270, 83, 83]" = torch.ops.aten.le.Scalar(relu_15, 0)
    where_170: "f32[8, 270, 83, 83]" = torch.ops.aten.where.self(le_170, full_default, add_1184);  add_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_67: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(add_1183, 1, 0, 108)
    slice_68: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(add_1183, 1, 108, 216)
    slice_69: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(add_1183, 1, 216, 324)
    slice_70: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(add_1183, 1, 324, 432)
    slice_71: "f32[8, 108, 42, 42]" = torch.ops.aten.slice.Tensor(add_1183, 1, 432, 540);  add_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_342: "f32[108]" = torch.ops.aten.sum.dim_IntList(slice_71, [0, 2, 3])
    sub_881: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_2846);  convolution_55 = unsqueeze_2846 = None
    mul_2937: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(slice_71, sub_881)
    sum_343: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2937, [0, 2, 3]);  mul_2937 = None
    mul_2938: "f32[108]" = torch.ops.aten.mul.Tensor(sum_342, 7.086167800453515e-05)
    unsqueeze_2847: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2938, 0);  mul_2938 = None
    unsqueeze_2848: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2847, 2);  unsqueeze_2847 = None
    unsqueeze_2849: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2848, 3);  unsqueeze_2848 = None
    mul_2939: "f32[108]" = torch.ops.aten.mul.Tensor(sum_343, 7.086167800453515e-05)
    mul_2940: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_2941: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2939, mul_2940);  mul_2939 = mul_2940 = None
    unsqueeze_2850: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2941, 0);  mul_2941 = None
    unsqueeze_2851: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2850, 2);  unsqueeze_2850 = None
    unsqueeze_2852: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2851, 3);  unsqueeze_2851 = None
    mul_2942: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_129);  primals_129 = None
    unsqueeze_2853: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2942, 0);  mul_2942 = None
    unsqueeze_2854: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2853, 2);  unsqueeze_2853 = None
    unsqueeze_2855: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2854, 3);  unsqueeze_2854 = None
    mul_2943: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_881, unsqueeze_2852);  sub_881 = unsqueeze_2852 = None
    sub_883: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(slice_71, mul_2943);  mul_2943 = None
    sub_884: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_883, unsqueeze_2849);  sub_883 = None
    mul_2944: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_884, unsqueeze_2855);  sub_884 = unsqueeze_2855 = None
    mul_2945: "f32[108]" = torch.ops.aten.mul.Tensor(sum_343, squeeze_91);  sum_343 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_317 = torch.ops.aten.convolution_backward.default(mul_2944, constant_pad_nd_18, primals_14, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2944 = constant_pad_nd_18 = primals_14 = None
    getitem_1437: "f32[8, 108, 83, 83]" = convolution_backward_317[0]
    getitem_1438: "f32[108, 108, 1, 1]" = convolution_backward_317[1];  convolution_backward_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_171: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_171, full_default, getitem_1437);  getitem_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_885: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_2858);  convolution_54 = unsqueeze_2858 = None
    mul_2946: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(slice_71, sub_885)
    sum_345: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2946, [0, 2, 3]);  mul_2946 = None
    mul_2948: "f32[108]" = torch.ops.aten.mul.Tensor(sum_345, 7.086167800453515e-05)
    mul_2949: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_2950: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2948, mul_2949);  mul_2948 = mul_2949 = None
    unsqueeze_2862: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2950, 0);  mul_2950 = None
    unsqueeze_2863: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2862, 2);  unsqueeze_2862 = None
    unsqueeze_2864: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2863, 3);  unsqueeze_2863 = None
    mul_2951: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_127);  primals_127 = None
    unsqueeze_2865: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2951, 0);  mul_2951 = None
    unsqueeze_2866: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2865, 2);  unsqueeze_2865 = None
    unsqueeze_2867: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2866, 3);  unsqueeze_2866 = None
    mul_2952: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_885, unsqueeze_2864);  sub_885 = unsqueeze_2864 = None
    sub_887: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(slice_71, mul_2952);  slice_71 = mul_2952 = None
    sub_888: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_887, unsqueeze_2849);  sub_887 = unsqueeze_2849 = None
    mul_2953: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_888, unsqueeze_2867);  sub_888 = unsqueeze_2867 = None
    mul_2954: "f32[108]" = torch.ops.aten.mul.Tensor(sum_345, squeeze_88);  sum_345 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_318 = torch.ops.aten.convolution_backward.default(mul_2953, convolution_53, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2953 = convolution_53 = primals_126 = None
    getitem_1440: "f32[8, 108, 42, 42]" = convolution_backward_318[0]
    getitem_1441: "f32[108, 108, 1, 1]" = convolution_backward_318[1];  convolution_backward_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_319 = torch.ops.aten.convolution_backward.default(getitem_1440, relu_27, primals_125, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1440 = primals_125 = None
    getitem_1443: "f32[8, 108, 42, 42]" = convolution_backward_319[0]
    getitem_1444: "f32[108, 1, 3, 3]" = convolution_backward_319[1];  convolution_backward_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_172: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_172: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_172, full_default, getitem_1443);  le_172 = getitem_1443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_346: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_172, [0, 2, 3])
    sub_889: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_2870);  convolution_52 = unsqueeze_2870 = None
    mul_2955: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_172, sub_889)
    sum_347: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2955, [0, 2, 3]);  mul_2955 = None
    mul_2956: "f32[108]" = torch.ops.aten.mul.Tensor(sum_346, 7.086167800453515e-05)
    unsqueeze_2871: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2956, 0);  mul_2956 = None
    unsqueeze_2872: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2871, 2);  unsqueeze_2871 = None
    unsqueeze_2873: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2872, 3);  unsqueeze_2872 = None
    mul_2957: "f32[108]" = torch.ops.aten.mul.Tensor(sum_347, 7.086167800453515e-05)
    mul_2958: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_2959: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2957, mul_2958);  mul_2957 = mul_2958 = None
    unsqueeze_2874: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2959, 0);  mul_2959 = None
    unsqueeze_2875: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2874, 2);  unsqueeze_2874 = None
    unsqueeze_2876: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2875, 3);  unsqueeze_2875 = None
    mul_2960: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_123);  primals_123 = None
    unsqueeze_2877: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2960, 0);  mul_2960 = None
    unsqueeze_2878: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2877, 2);  unsqueeze_2877 = None
    unsqueeze_2879: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2878, 3);  unsqueeze_2878 = None
    mul_2961: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_889, unsqueeze_2876);  sub_889 = unsqueeze_2876 = None
    sub_891: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_172, mul_2961);  where_172 = mul_2961 = None
    sub_892: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_891, unsqueeze_2873);  sub_891 = unsqueeze_2873 = None
    mul_2962: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_892, unsqueeze_2879);  sub_892 = unsqueeze_2879 = None
    mul_2963: "f32[108]" = torch.ops.aten.mul.Tensor(sum_347, squeeze_85);  sum_347 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_320 = torch.ops.aten.convolution_backward.default(mul_2962, convolution_51, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2962 = convolution_51 = primals_122 = None
    getitem_1446: "f32[8, 108, 42, 42]" = convolution_backward_320[0]
    getitem_1447: "f32[108, 108, 1, 1]" = convolution_backward_320[1];  convolution_backward_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_321 = torch.ops.aten.convolution_backward.default(getitem_1446, constant_pad_nd_17, primals_13, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1446 = constant_pad_nd_17 = primals_13 = None
    getitem_1449: "f32[8, 108, 85, 85]" = convolution_backward_321[0]
    getitem_1450: "f32[108, 1, 3, 3]" = convolution_backward_321[1];  convolution_backward_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_62: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(getitem_1449, [-1, -1, -1, -1]);  getitem_1449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_173: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_173, full_default, constant_pad_nd_62);  constant_pad_nd_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_36: "f32[8, 108, 85, 85]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_70, constant_pad_nd_13, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_63: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_36, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1185: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(where_171, constant_pad_nd_63);  where_171 = constant_pad_nd_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_348: "f32[108]" = torch.ops.aten.sum.dim_IntList(slice_70, [0, 2, 3])
    sub_893: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_2882);  convolution_50 = unsqueeze_2882 = None
    mul_2964: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(slice_70, sub_893)
    sum_349: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2964, [0, 2, 3]);  mul_2964 = None
    mul_2965: "f32[108]" = torch.ops.aten.mul.Tensor(sum_348, 7.086167800453515e-05)
    unsqueeze_2883: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2965, 0);  mul_2965 = None
    unsqueeze_2884: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2883, 2);  unsqueeze_2883 = None
    unsqueeze_2885: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2884, 3);  unsqueeze_2884 = None
    mul_2966: "f32[108]" = torch.ops.aten.mul.Tensor(sum_349, 7.086167800453515e-05)
    mul_2967: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_2968: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2966, mul_2967);  mul_2966 = mul_2967 = None
    unsqueeze_2886: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2968, 0);  mul_2968 = None
    unsqueeze_2887: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2886, 2);  unsqueeze_2886 = None
    unsqueeze_2888: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2887, 3);  unsqueeze_2887 = None
    mul_2969: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_120);  primals_120 = None
    unsqueeze_2889: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2969, 0);  mul_2969 = None
    unsqueeze_2890: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2889, 2);  unsqueeze_2889 = None
    unsqueeze_2891: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2890, 3);  unsqueeze_2890 = None
    mul_2970: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_893, unsqueeze_2888);  sub_893 = unsqueeze_2888 = None
    sub_895: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(slice_70, mul_2970);  slice_70 = mul_2970 = None
    sub_896: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_895, unsqueeze_2885);  sub_895 = unsqueeze_2885 = None
    mul_2971: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_896, unsqueeze_2891);  sub_896 = unsqueeze_2891 = None
    mul_2972: "f32[108]" = torch.ops.aten.mul.Tensor(sum_349, squeeze_82);  sum_349 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_322 = torch.ops.aten.convolution_backward.default(mul_2971, convolution_49, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2971 = convolution_49 = primals_119 = None
    getitem_1452: "f32[8, 108, 42, 42]" = convolution_backward_322[0]
    getitem_1453: "f32[108, 108, 1, 1]" = convolution_backward_322[1];  convolution_backward_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_323 = torch.ops.aten.convolution_backward.default(getitem_1452, relu_25, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1452 = primals_118 = None
    getitem_1455: "f32[8, 108, 42, 42]" = convolution_backward_323[0]
    getitem_1456: "f32[108, 1, 3, 3]" = convolution_backward_323[1];  convolution_backward_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_174: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_174: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_174, full_default, getitem_1455);  le_174 = getitem_1455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_350: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_174, [0, 2, 3])
    sub_897: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_2894);  convolution_48 = unsqueeze_2894 = None
    mul_2973: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_174, sub_897)
    sum_351: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2973, [0, 2, 3]);  mul_2973 = None
    mul_2974: "f32[108]" = torch.ops.aten.mul.Tensor(sum_350, 7.086167800453515e-05)
    unsqueeze_2895: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2974, 0);  mul_2974 = None
    unsqueeze_2896: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2895, 2);  unsqueeze_2895 = None
    unsqueeze_2897: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2896, 3);  unsqueeze_2896 = None
    mul_2975: "f32[108]" = torch.ops.aten.mul.Tensor(sum_351, 7.086167800453515e-05)
    mul_2976: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_2977: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2975, mul_2976);  mul_2975 = mul_2976 = None
    unsqueeze_2898: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2977, 0);  mul_2977 = None
    unsqueeze_2899: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2898, 2);  unsqueeze_2898 = None
    unsqueeze_2900: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2899, 3);  unsqueeze_2899 = None
    mul_2978: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_116);  primals_116 = None
    unsqueeze_2901: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2978, 0);  mul_2978 = None
    unsqueeze_2902: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2901, 2);  unsqueeze_2901 = None
    unsqueeze_2903: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2902, 3);  unsqueeze_2902 = None
    mul_2979: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_897, unsqueeze_2900);  sub_897 = unsqueeze_2900 = None
    sub_899: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_174, mul_2979);  where_174 = mul_2979 = None
    sub_900: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_899, unsqueeze_2897);  sub_899 = unsqueeze_2897 = None
    mul_2980: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_900, unsqueeze_2903);  sub_900 = unsqueeze_2903 = None
    mul_2981: "f32[108]" = torch.ops.aten.mul.Tensor(sum_351, squeeze_79);  sum_351 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_324 = torch.ops.aten.convolution_backward.default(mul_2980, convolution_47, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2980 = convolution_47 = primals_115 = None
    getitem_1458: "f32[8, 108, 42, 42]" = convolution_backward_324[0]
    getitem_1459: "f32[108, 108, 1, 1]" = convolution_backward_324[1];  convolution_backward_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_325 = torch.ops.aten.convolution_backward.default(getitem_1458, relu_24, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1458 = primals_114 = None
    getitem_1461: "f32[8, 108, 42, 42]" = convolution_backward_325[0]
    getitem_1462: "f32[108, 1, 3, 3]" = convolution_backward_325[1];  convolution_backward_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_175: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_175: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_175, full_default, getitem_1461);  le_175 = getitem_1461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1186: "f32[8, 108, 42, 42]" = torch.ops.aten.add.Tensor(slice_69, where_175);  slice_69 = where_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_352: "f32[108]" = torch.ops.aten.sum.dim_IntList(add_1186, [0, 2, 3])
    sub_901: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_2906);  convolution_46 = unsqueeze_2906 = None
    mul_2982: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(add_1186, sub_901)
    sum_353: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2982, [0, 2, 3]);  mul_2982 = None
    mul_2983: "f32[108]" = torch.ops.aten.mul.Tensor(sum_352, 7.086167800453515e-05)
    unsqueeze_2907: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2983, 0);  mul_2983 = None
    unsqueeze_2908: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2907, 2);  unsqueeze_2907 = None
    unsqueeze_2909: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2908, 3);  unsqueeze_2908 = None
    mul_2984: "f32[108]" = torch.ops.aten.mul.Tensor(sum_353, 7.086167800453515e-05)
    mul_2985: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_2986: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2984, mul_2985);  mul_2984 = mul_2985 = None
    unsqueeze_2910: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2986, 0);  mul_2986 = None
    unsqueeze_2911: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2910, 2);  unsqueeze_2910 = None
    unsqueeze_2912: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2911, 3);  unsqueeze_2911 = None
    mul_2987: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_112);  primals_112 = None
    unsqueeze_2913: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2987, 0);  mul_2987 = None
    unsqueeze_2914: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2913, 2);  unsqueeze_2913 = None
    unsqueeze_2915: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2914, 3);  unsqueeze_2914 = None
    mul_2988: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_901, unsqueeze_2912);  sub_901 = unsqueeze_2912 = None
    sub_903: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(add_1186, mul_2988);  mul_2988 = None
    sub_904: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_903, unsqueeze_2909);  sub_903 = None
    mul_2989: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_904, unsqueeze_2915);  sub_904 = unsqueeze_2915 = None
    mul_2990: "f32[108]" = torch.ops.aten.mul.Tensor(sum_353, squeeze_76);  sum_353 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_326 = torch.ops.aten.convolution_backward.default(mul_2989, convolution_45, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2989 = convolution_45 = primals_111 = None
    getitem_1464: "f32[8, 108, 42, 42]" = convolution_backward_326[0]
    getitem_1465: "f32[108, 108, 1, 1]" = convolution_backward_326[1];  convolution_backward_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_327 = torch.ops.aten.convolution_backward.default(getitem_1464, relu_23, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1464 = primals_110 = None
    getitem_1467: "f32[8, 108, 42, 42]" = convolution_backward_327[0]
    getitem_1468: "f32[108, 1, 3, 3]" = convolution_backward_327[1];  convolution_backward_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_176: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_176: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_176, full_default, getitem_1467);  le_176 = getitem_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_354: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_176, [0, 2, 3])
    sub_905: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_2918);  convolution_44 = unsqueeze_2918 = None
    mul_2991: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_176, sub_905)
    sum_355: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_2991, [0, 2, 3]);  mul_2991 = None
    mul_2992: "f32[108]" = torch.ops.aten.mul.Tensor(sum_354, 7.086167800453515e-05)
    unsqueeze_2919: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2992, 0);  mul_2992 = None
    unsqueeze_2920: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2919, 2);  unsqueeze_2919 = None
    unsqueeze_2921: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2920, 3);  unsqueeze_2920 = None
    mul_2993: "f32[108]" = torch.ops.aten.mul.Tensor(sum_355, 7.086167800453515e-05)
    mul_2994: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_2995: "f32[108]" = torch.ops.aten.mul.Tensor(mul_2993, mul_2994);  mul_2993 = mul_2994 = None
    unsqueeze_2922: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2995, 0);  mul_2995 = None
    unsqueeze_2923: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2922, 2);  unsqueeze_2922 = None
    unsqueeze_2924: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2923, 3);  unsqueeze_2923 = None
    mul_2996: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_108);  primals_108 = None
    unsqueeze_2925: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_2996, 0);  mul_2996 = None
    unsqueeze_2926: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2925, 2);  unsqueeze_2925 = None
    unsqueeze_2927: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2926, 3);  unsqueeze_2926 = None
    mul_2997: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_905, unsqueeze_2924);  sub_905 = unsqueeze_2924 = None
    sub_907: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_176, mul_2997);  where_176 = mul_2997 = None
    sub_908: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_907, unsqueeze_2921);  sub_907 = unsqueeze_2921 = None
    mul_2998: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_908, unsqueeze_2927);  sub_908 = unsqueeze_2927 = None
    mul_2999: "f32[108]" = torch.ops.aten.mul.Tensor(sum_355, squeeze_73);  sum_355 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_328 = torch.ops.aten.convolution_backward.default(mul_2998, convolution_43, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2998 = convolution_43 = primals_107 = None
    getitem_1470: "f32[8, 108, 42, 42]" = convolution_backward_328[0]
    getitem_1471: "f32[108, 108, 1, 1]" = convolution_backward_328[1];  convolution_backward_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_329 = torch.ops.aten.convolution_backward.default(getitem_1470, constant_pad_nd_15, primals_12, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1470 = constant_pad_nd_15 = primals_12 = None
    getitem_1473: "f32[8, 108, 85, 85]" = convolution_backward_329[0]
    getitem_1474: "f32[108, 1, 3, 3]" = convolution_backward_329[1];  convolution_backward_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_64: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(getitem_1473, [-1, -1, -1, -1]);  getitem_1473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_177: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_171, full_default, constant_pad_nd_64);  constant_pad_nd_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1187: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(add_1185, where_177);  add_1185 = where_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_909: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_2930);  convolution_42 = unsqueeze_2930 = None
    mul_3000: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(add_1186, sub_909)
    sum_357: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3000, [0, 2, 3]);  mul_3000 = None
    mul_3002: "f32[108]" = torch.ops.aten.mul.Tensor(sum_357, 7.086167800453515e-05)
    mul_3003: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_3004: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3002, mul_3003);  mul_3002 = mul_3003 = None
    unsqueeze_2934: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3004, 0);  mul_3004 = None
    unsqueeze_2935: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2934, 2);  unsqueeze_2934 = None
    unsqueeze_2936: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2935, 3);  unsqueeze_2935 = None
    mul_3005: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_105);  primals_105 = None
    unsqueeze_2937: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3005, 0);  mul_3005 = None
    unsqueeze_2938: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2937, 2);  unsqueeze_2937 = None
    unsqueeze_2939: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2938, 3);  unsqueeze_2938 = None
    mul_3006: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_909, unsqueeze_2936);  sub_909 = unsqueeze_2936 = None
    sub_911: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(add_1186, mul_3006);  add_1186 = mul_3006 = None
    sub_912: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_911, unsqueeze_2909);  sub_911 = unsqueeze_2909 = None
    mul_3007: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_912, unsqueeze_2939);  sub_912 = unsqueeze_2939 = None
    mul_3008: "f32[108]" = torch.ops.aten.mul.Tensor(sum_357, squeeze_70);  sum_357 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_330 = torch.ops.aten.convolution_backward.default(mul_3007, convolution_41, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3007 = convolution_41 = primals_104 = None
    getitem_1476: "f32[8, 108, 42, 42]" = convolution_backward_330[0]
    getitem_1477: "f32[108, 108, 1, 1]" = convolution_backward_330[1];  convolution_backward_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_331 = torch.ops.aten.convolution_backward.default(getitem_1476, relu_21, primals_103, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1476 = primals_103 = None
    getitem_1479: "f32[8, 108, 42, 42]" = convolution_backward_331[0]
    getitem_1480: "f32[108, 1, 5, 5]" = convolution_backward_331[1];  convolution_backward_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_178: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_178: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_178, full_default, getitem_1479);  le_178 = getitem_1479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_358: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_178, [0, 2, 3])
    sub_913: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_2942);  convolution_40 = unsqueeze_2942 = None
    mul_3009: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_178, sub_913)
    sum_359: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3009, [0, 2, 3]);  mul_3009 = None
    mul_3010: "f32[108]" = torch.ops.aten.mul.Tensor(sum_358, 7.086167800453515e-05)
    unsqueeze_2943: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3010, 0);  mul_3010 = None
    unsqueeze_2944: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2943, 2);  unsqueeze_2943 = None
    unsqueeze_2945: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2944, 3);  unsqueeze_2944 = None
    mul_3011: "f32[108]" = torch.ops.aten.mul.Tensor(sum_359, 7.086167800453515e-05)
    mul_3012: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_3013: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3011, mul_3012);  mul_3011 = mul_3012 = None
    unsqueeze_2946: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3013, 0);  mul_3013 = None
    unsqueeze_2947: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2946, 2);  unsqueeze_2946 = None
    unsqueeze_2948: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2947, 3);  unsqueeze_2947 = None
    mul_3014: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_101);  primals_101 = None
    unsqueeze_2949: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3014, 0);  mul_3014 = None
    unsqueeze_2950: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2949, 2);  unsqueeze_2949 = None
    unsqueeze_2951: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2950, 3);  unsqueeze_2950 = None
    mul_3015: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_913, unsqueeze_2948);  sub_913 = unsqueeze_2948 = None
    sub_915: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_178, mul_3015);  where_178 = mul_3015 = None
    sub_916: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_915, unsqueeze_2945);  sub_915 = unsqueeze_2945 = None
    mul_3016: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_916, unsqueeze_2951);  sub_916 = unsqueeze_2951 = None
    mul_3017: "f32[108]" = torch.ops.aten.mul.Tensor(sum_359, squeeze_67);  sum_359 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_332 = torch.ops.aten.convolution_backward.default(mul_3016, convolution_39, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3016 = convolution_39 = primals_100 = None
    getitem_1482: "f32[8, 108, 42, 42]" = convolution_backward_332[0]
    getitem_1483: "f32[108, 108, 1, 1]" = convolution_backward_332[1];  convolution_backward_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_333 = torch.ops.aten.convolution_backward.default(getitem_1482, constant_pad_nd_14, primals_11, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1482 = constant_pad_nd_14 = primals_11 = None
    getitem_1485: "f32[8, 108, 87, 87]" = convolution_backward_333[0]
    getitem_1486: "f32[108, 1, 5, 5]" = convolution_backward_333[1];  convolution_backward_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_65: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(getitem_1485, [-2, -2, -2, -2]);  getitem_1485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_179: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_171, full_default, constant_pad_nd_65);  constant_pad_nd_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1188: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(add_1187, where_179);  add_1187 = where_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_37: "f32[8, 108, 85, 85]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_68, constant_pad_nd_13, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_53);  constant_pad_nd_13 = getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_66: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_37, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1189: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(add_1188, constant_pad_nd_66);  add_1188 = constant_pad_nd_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_360: "f32[108]" = torch.ops.aten.sum.dim_IntList(slice_68, [0, 2, 3])
    sub_917: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_2954);  convolution_38 = unsqueeze_2954 = None
    mul_3018: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(slice_68, sub_917)
    sum_361: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3018, [0, 2, 3]);  mul_3018 = None
    mul_3019: "f32[108]" = torch.ops.aten.mul.Tensor(sum_360, 7.086167800453515e-05)
    unsqueeze_2955: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3019, 0);  mul_3019 = None
    unsqueeze_2956: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2955, 2);  unsqueeze_2955 = None
    unsqueeze_2957: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2956, 3);  unsqueeze_2956 = None
    mul_3020: "f32[108]" = torch.ops.aten.mul.Tensor(sum_361, 7.086167800453515e-05)
    mul_3021: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_3022: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3020, mul_3021);  mul_3020 = mul_3021 = None
    unsqueeze_2958: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3022, 0);  mul_3022 = None
    unsqueeze_2959: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2958, 2);  unsqueeze_2958 = None
    unsqueeze_2960: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2959, 3);  unsqueeze_2959 = None
    mul_3023: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_98);  primals_98 = None
    unsqueeze_2961: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3023, 0);  mul_3023 = None
    unsqueeze_2962: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2961, 2);  unsqueeze_2961 = None
    unsqueeze_2963: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2962, 3);  unsqueeze_2962 = None
    mul_3024: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_917, unsqueeze_2960);  sub_917 = unsqueeze_2960 = None
    sub_919: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(slice_68, mul_3024);  slice_68 = mul_3024 = None
    sub_920: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_919, unsqueeze_2957);  sub_919 = unsqueeze_2957 = None
    mul_3025: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_920, unsqueeze_2963);  sub_920 = unsqueeze_2963 = None
    mul_3026: "f32[108]" = torch.ops.aten.mul.Tensor(sum_361, squeeze_64);  sum_361 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_334 = torch.ops.aten.convolution_backward.default(mul_3025, convolution_37, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3025 = convolution_37 = primals_97 = None
    getitem_1488: "f32[8, 108, 42, 42]" = convolution_backward_334[0]
    getitem_1489: "f32[108, 108, 1, 1]" = convolution_backward_334[1];  convolution_backward_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_335 = torch.ops.aten.convolution_backward.default(getitem_1488, relu_19, primals_96, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1488 = primals_96 = None
    getitem_1491: "f32[8, 108, 42, 42]" = convolution_backward_335[0]
    getitem_1492: "f32[108, 1, 7, 7]" = convolution_backward_335[1];  convolution_backward_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_180: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_180: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_180, full_default, getitem_1491);  le_180 = getitem_1491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_362: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_180, [0, 2, 3])
    sub_921: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_2966);  convolution_36 = unsqueeze_2966 = None
    mul_3027: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_180, sub_921)
    sum_363: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3027, [0, 2, 3]);  mul_3027 = None
    mul_3028: "f32[108]" = torch.ops.aten.mul.Tensor(sum_362, 7.086167800453515e-05)
    unsqueeze_2967: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3028, 0);  mul_3028 = None
    unsqueeze_2968: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2967, 2);  unsqueeze_2967 = None
    unsqueeze_2969: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2968, 3);  unsqueeze_2968 = None
    mul_3029: "f32[108]" = torch.ops.aten.mul.Tensor(sum_363, 7.086167800453515e-05)
    mul_3030: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_3031: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3029, mul_3030);  mul_3029 = mul_3030 = None
    unsqueeze_2970: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3031, 0);  mul_3031 = None
    unsqueeze_2971: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2970, 2);  unsqueeze_2970 = None
    unsqueeze_2972: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2971, 3);  unsqueeze_2971 = None
    mul_3032: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_94);  primals_94 = None
    unsqueeze_2973: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3032, 0);  mul_3032 = None
    unsqueeze_2974: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2973, 2);  unsqueeze_2973 = None
    unsqueeze_2975: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2974, 3);  unsqueeze_2974 = None
    mul_3033: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_921, unsqueeze_2972);  sub_921 = unsqueeze_2972 = None
    sub_923: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_180, mul_3033);  where_180 = mul_3033 = None
    sub_924: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_923, unsqueeze_2969);  sub_923 = unsqueeze_2969 = None
    mul_3034: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_924, unsqueeze_2975);  sub_924 = unsqueeze_2975 = None
    mul_3035: "f32[108]" = torch.ops.aten.mul.Tensor(sum_363, squeeze_61);  sum_363 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_336 = torch.ops.aten.convolution_backward.default(mul_3034, convolution_35, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3034 = convolution_35 = primals_93 = None
    getitem_1494: "f32[8, 108, 42, 42]" = convolution_backward_336[0]
    getitem_1495: "f32[108, 108, 1, 1]" = convolution_backward_336[1];  convolution_backward_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_337 = torch.ops.aten.convolution_backward.default(getitem_1494, constant_pad_nd_12, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1494 = constant_pad_nd_12 = primals_10 = None
    getitem_1497: "f32[8, 108, 89, 89]" = convolution_backward_337[0]
    getitem_1498: "f32[108, 1, 7, 7]" = convolution_backward_337[1];  convolution_backward_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_67: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(getitem_1497, [-3, -3, -3, -3]);  getitem_1497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_181: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_171, full_default, constant_pad_nd_67);  le_171 = constant_pad_nd_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1190: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(add_1189, where_181);  add_1189 = where_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_38: "f32[8, 108, 85, 85]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_67, constant_pad_nd_11, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_47);  constant_pad_nd_11 = getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_68: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_38, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1191: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(where_173, constant_pad_nd_68);  where_173 = constant_pad_nd_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_364: "f32[108]" = torch.ops.aten.sum.dim_IntList(slice_67, [0, 2, 3])
    sub_925: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_2978);  convolution_34 = unsqueeze_2978 = None
    mul_3036: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(slice_67, sub_925)
    sum_365: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3036, [0, 2, 3]);  mul_3036 = None
    mul_3037: "f32[108]" = torch.ops.aten.mul.Tensor(sum_364, 7.086167800453515e-05)
    unsqueeze_2979: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3037, 0);  mul_3037 = None
    unsqueeze_2980: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2979, 2);  unsqueeze_2979 = None
    unsqueeze_2981: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2980, 3);  unsqueeze_2980 = None
    mul_3038: "f32[108]" = torch.ops.aten.mul.Tensor(sum_365, 7.086167800453515e-05)
    mul_3039: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_3040: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3038, mul_3039);  mul_3038 = mul_3039 = None
    unsqueeze_2982: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3040, 0);  mul_3040 = None
    unsqueeze_2983: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2982, 2);  unsqueeze_2982 = None
    unsqueeze_2984: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2983, 3);  unsqueeze_2983 = None
    mul_3041: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_91);  primals_91 = None
    unsqueeze_2985: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3041, 0);  mul_3041 = None
    unsqueeze_2986: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2985, 2);  unsqueeze_2985 = None
    unsqueeze_2987: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2986, 3);  unsqueeze_2986 = None
    mul_3042: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_925, unsqueeze_2984);  sub_925 = unsqueeze_2984 = None
    sub_927: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(slice_67, mul_3042);  slice_67 = mul_3042 = None
    sub_928: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_927, unsqueeze_2981);  sub_927 = unsqueeze_2981 = None
    mul_3043: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_928, unsqueeze_2987);  sub_928 = unsqueeze_2987 = None
    mul_3044: "f32[108]" = torch.ops.aten.mul.Tensor(sum_365, squeeze_58);  sum_365 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_338 = torch.ops.aten.convolution_backward.default(mul_3043, convolution_33, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3043 = convolution_33 = primals_90 = None
    getitem_1500: "f32[8, 108, 42, 42]" = convolution_backward_338[0]
    getitem_1501: "f32[108, 108, 1, 1]" = convolution_backward_338[1];  convolution_backward_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_339 = torch.ops.aten.convolution_backward.default(getitem_1500, relu_17, primals_89, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1500 = primals_89 = None
    getitem_1503: "f32[8, 108, 42, 42]" = convolution_backward_339[0]
    getitem_1504: "f32[108, 1, 5, 5]" = convolution_backward_339[1];  convolution_backward_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_182: "b8[8, 108, 42, 42]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_182: "f32[8, 108, 42, 42]" = torch.ops.aten.where.self(le_182, full_default, getitem_1503);  le_182 = getitem_1503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_366: "f32[108]" = torch.ops.aten.sum.dim_IntList(where_182, [0, 2, 3])
    sub_929: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_2990);  convolution_32 = unsqueeze_2990 = None
    mul_3045: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(where_182, sub_929)
    sum_367: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3045, [0, 2, 3]);  mul_3045 = None
    mul_3046: "f32[108]" = torch.ops.aten.mul.Tensor(sum_366, 7.086167800453515e-05)
    unsqueeze_2991: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3046, 0);  mul_3046 = None
    unsqueeze_2992: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2991, 2);  unsqueeze_2991 = None
    unsqueeze_2993: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2992, 3);  unsqueeze_2992 = None
    mul_3047: "f32[108]" = torch.ops.aten.mul.Tensor(sum_367, 7.086167800453515e-05)
    mul_3048: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_3049: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3047, mul_3048);  mul_3047 = mul_3048 = None
    unsqueeze_2994: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3049, 0);  mul_3049 = None
    unsqueeze_2995: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2994, 2);  unsqueeze_2994 = None
    unsqueeze_2996: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2995, 3);  unsqueeze_2995 = None
    mul_3050: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_87);  primals_87 = None
    unsqueeze_2997: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3050, 0);  mul_3050 = None
    unsqueeze_2998: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2997, 2);  unsqueeze_2997 = None
    unsqueeze_2999: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2998, 3);  unsqueeze_2998 = None
    mul_3051: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_929, unsqueeze_2996);  sub_929 = unsqueeze_2996 = None
    sub_931: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(where_182, mul_3051);  where_182 = mul_3051 = None
    sub_932: "f32[8, 108, 42, 42]" = torch.ops.aten.sub.Tensor(sub_931, unsqueeze_2993);  sub_931 = unsqueeze_2993 = None
    mul_3052: "f32[8, 108, 42, 42]" = torch.ops.aten.mul.Tensor(sub_932, unsqueeze_2999);  sub_932 = unsqueeze_2999 = None
    mul_3053: "f32[108]" = torch.ops.aten.mul.Tensor(sum_367, squeeze_55);  sum_367 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_340 = torch.ops.aten.convolution_backward.default(mul_3052, convolution_31, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3052 = convolution_31 = primals_86 = None
    getitem_1506: "f32[8, 108, 42, 42]" = convolution_backward_340[0]
    getitem_1507: "f32[108, 108, 1, 1]" = convolution_backward_340[1];  convolution_backward_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_341 = torch.ops.aten.convolution_backward.default(getitem_1506, constant_pad_nd_10, primals_9, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 108, [True, True, False]);  getitem_1506 = constant_pad_nd_10 = primals_9 = None
    getitem_1509: "f32[8, 108, 87, 87]" = convolution_backward_341[0]
    getitem_1510: "f32[108, 1, 5, 5]" = convolution_backward_341[1];  convolution_backward_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_69: "f32[8, 108, 83, 83]" = torch.ops.aten.constant_pad_nd.default(getitem_1509, [-2, -2, -2, -2]);  getitem_1509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_183: "f32[8, 108, 83, 83]" = torch.ops.aten.where.self(le_173, full_default, constant_pad_nd_69);  le_173 = constant_pad_nd_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1192: "f32[8, 108, 83, 83]" = torch.ops.aten.add.Tensor(add_1191, where_183);  add_1191 = where_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_368: "f32[108]" = torch.ops.aten.sum.dim_IntList(add_1190, [0, 2, 3])
    sub_933: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_3002);  convolution_30 = unsqueeze_3002 = None
    mul_3054: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(add_1190, sub_933)
    sum_369: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3054, [0, 2, 3]);  mul_3054 = None
    mul_3055: "f32[108]" = torch.ops.aten.mul.Tensor(sum_368, 1.814486863115111e-05)
    unsqueeze_3003: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3055, 0);  mul_3055 = None
    unsqueeze_3004: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3003, 2);  unsqueeze_3003 = None
    unsqueeze_3005: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3004, 3);  unsqueeze_3004 = None
    mul_3056: "f32[108]" = torch.ops.aten.mul.Tensor(sum_369, 1.814486863115111e-05)
    mul_3057: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_3058: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3056, mul_3057);  mul_3056 = mul_3057 = None
    unsqueeze_3006: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3058, 0);  mul_3058 = None
    unsqueeze_3007: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3006, 2);  unsqueeze_3006 = None
    unsqueeze_3008: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3007, 3);  unsqueeze_3007 = None
    mul_3059: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_84);  primals_84 = None
    unsqueeze_3009: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3059, 0);  mul_3059 = None
    unsqueeze_3010: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3009, 2);  unsqueeze_3009 = None
    unsqueeze_3011: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3010, 3);  unsqueeze_3010 = None
    mul_3060: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_933, unsqueeze_3008);  sub_933 = unsqueeze_3008 = None
    sub_935: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(add_1190, mul_3060);  add_1190 = mul_3060 = None
    sub_936: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(sub_935, unsqueeze_3005);  sub_935 = unsqueeze_3005 = None
    mul_3061: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_936, unsqueeze_3011);  sub_936 = unsqueeze_3011 = None
    mul_3062: "f32[108]" = torch.ops.aten.mul.Tensor(sum_369, squeeze_52);  sum_369 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_342 = torch.ops.aten.convolution_backward.default(mul_3061, relu_15, primals_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3061 = relu_15 = primals_83 = None
    getitem_1512: "f32[8, 270, 83, 83]" = convolution_backward_342[0]
    getitem_1513: "f32[108, 270, 1, 1]" = convolution_backward_342[1];  convolution_backward_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_184: "f32[8, 270, 83, 83]" = torch.ops.aten.where.self(le_170, full_default, getitem_1512);  le_170 = getitem_1512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1193: "f32[8, 270, 83, 83]" = torch.ops.aten.add.Tensor(where_170, where_184);  where_170 = where_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:98, code: out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    sum_370: "f32[108]" = torch.ops.aten.sum.dim_IntList(add_1192, [0, 2, 3])
    sub_937: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_3014);  cat_1 = unsqueeze_3014 = None
    mul_3063: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(add_1192, sub_937)
    sum_371: "f32[108]" = torch.ops.aten.sum.dim_IntList(mul_3063, [0, 2, 3]);  mul_3063 = None
    mul_3064: "f32[108]" = torch.ops.aten.mul.Tensor(sum_370, 1.814486863115111e-05)
    unsqueeze_3015: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3064, 0);  mul_3064 = None
    unsqueeze_3016: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3015, 2);  unsqueeze_3015 = None
    unsqueeze_3017: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3016, 3);  unsqueeze_3016 = None
    mul_3065: "f32[108]" = torch.ops.aten.mul.Tensor(sum_371, 1.814486863115111e-05)
    mul_3066: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_3067: "f32[108]" = torch.ops.aten.mul.Tensor(mul_3065, mul_3066);  mul_3065 = mul_3066 = None
    unsqueeze_3018: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3067, 0);  mul_3067 = None
    unsqueeze_3019: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3018, 2);  unsqueeze_3018 = None
    unsqueeze_3020: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3019, 3);  unsqueeze_3019 = None
    mul_3068: "f32[108]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_81);  primals_81 = None
    unsqueeze_3021: "f32[1, 108]" = torch.ops.aten.unsqueeze.default(mul_3068, 0);  mul_3068 = None
    unsqueeze_3022: "f32[1, 108, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3021, 2);  unsqueeze_3021 = None
    unsqueeze_3023: "f32[1, 108, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3022, 3);  unsqueeze_3022 = None
    mul_3069: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_937, unsqueeze_3020);  sub_937 = unsqueeze_3020 = None
    sub_939: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(add_1192, mul_3069);  add_1192 = mul_3069 = None
    sub_940: "f32[8, 108, 83, 83]" = torch.ops.aten.sub.Tensor(sub_939, unsqueeze_3017);  sub_939 = unsqueeze_3017 = None
    mul_3070: "f32[8, 108, 83, 83]" = torch.ops.aten.mul.Tensor(sub_940, unsqueeze_3023);  sub_940 = unsqueeze_3023 = None
    mul_3071: "f32[108]" = torch.ops.aten.mul.Tensor(sum_371, squeeze_49);  sum_371 = squeeze_49 = None
    slice_72: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(mul_3070, 1, 0, 54)
    slice_73: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(mul_3070, 1, 54, 108);  mul_3070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:97, code: x_path2 = self.path_2(x)
    convolution_backward_343 = torch.ops.aten.convolution_backward.default(slice_73, avg_pool2d_1, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_73 = avg_pool2d_1 = primals_80 = None
    getitem_1515: "f32[8, 96, 83, 83]" = convolution_backward_343[0]
    getitem_1516: "f32[54, 96, 1, 1]" = convolution_backward_343[1];  convolution_backward_343 = None
    avg_pool2d_backward_6: "f32[8, 96, 165, 165]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1515, constant_pad_nd_9, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1515 = constant_pad_nd_9 = None
    constant_pad_nd_70: "f32[8, 96, 165, 165]" = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_6, [1, -1, 1, -1]);  avg_pool2d_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    convolution_backward_344 = torch.ops.aten.convolution_backward.default(slice_72, avg_pool2d, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_72 = avg_pool2d = primals_79 = None
    getitem_1518: "f32[8, 96, 83, 83]" = convolution_backward_344[0]
    getitem_1519: "f32[54, 96, 1, 1]" = convolution_backward_344[1];  convolution_backward_344 = None
    avg_pool2d_backward_7: "f32[8, 96, 165, 165]" = torch.ops.aten.avg_pool2d_backward.default(getitem_1518, relu, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_1518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:96, code: x_path1 = self.path_1(x)
    add_1194: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(constant_pad_nd_70, avg_pool2d_backward_7);  constant_pad_nd_70 = avg_pool2d_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:95, code: x = self.act(x)
    le_185: "b8[8, 96, 165, 165]" = torch.ops.aten.le.Scalar(relu, 0)
    where_185: "f32[8, 96, 165, 165]" = torch.ops.aten.where.self(le_185, full_default, add_1194);  add_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:128, code: x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    slice_74: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(add_1193, 1, 0, 54)
    slice_75: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(add_1193, 1, 54, 108)
    slice_76: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(add_1193, 1, 108, 162)
    slice_77: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(add_1193, 1, 162, 216)
    slice_78: "f32[8, 54, 83, 83]" = torch.ops.aten.slice.Tensor(add_1193, 1, 216, 270);  add_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_372: "f32[54]" = torch.ops.aten.sum.dim_IntList(slice_78, [0, 2, 3])
    sub_941: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_3026);  convolution_27 = unsqueeze_3026 = None
    mul_3072: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_78, sub_941)
    sum_373: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3072, [0, 2, 3]);  mul_3072 = None
    mul_3073: "f32[54]" = torch.ops.aten.mul.Tensor(sum_372, 1.814486863115111e-05)
    unsqueeze_3027: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3073, 0);  mul_3073 = None
    unsqueeze_3028: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3027, 2);  unsqueeze_3027 = None
    unsqueeze_3029: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3028, 3);  unsqueeze_3028 = None
    mul_3074: "f32[54]" = torch.ops.aten.mul.Tensor(sum_373, 1.814486863115111e-05)
    mul_3075: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_3076: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3074, mul_3075);  mul_3074 = mul_3075 = None
    unsqueeze_3030: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3076, 0);  mul_3076 = None
    unsqueeze_3031: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3030, 2);  unsqueeze_3030 = None
    unsqueeze_3032: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3031, 3);  unsqueeze_3031 = None
    mul_3077: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_77);  primals_77 = None
    unsqueeze_3033: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3077, 0);  mul_3077 = None
    unsqueeze_3034: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3033, 2);  unsqueeze_3033 = None
    unsqueeze_3035: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3034, 3);  unsqueeze_3034 = None
    mul_3078: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_941, unsqueeze_3032);  sub_941 = unsqueeze_3032 = None
    sub_943: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_78, mul_3078);  mul_3078 = None
    sub_944: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_943, unsqueeze_3029);  sub_943 = None
    mul_3079: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_944, unsqueeze_3035);  sub_944 = unsqueeze_3035 = None
    mul_3080: "f32[54]" = torch.ops.aten.mul.Tensor(sum_373, squeeze_46);  sum_373 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_345 = torch.ops.aten.convolution_backward.default(mul_3079, constant_pad_nd_8, primals_8, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3079 = constant_pad_nd_8 = primals_8 = None
    getitem_1521: "f32[8, 54, 165, 165]" = convolution_backward_345[0]
    getitem_1522: "f32[54, 54, 1, 1]" = convolution_backward_345[1];  convolution_backward_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_186: "f32[8, 54, 165, 165]" = torch.ops.aten.where.self(le_186, full_default, getitem_1521);  getitem_1521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_945: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_3038);  convolution_26 = unsqueeze_3038 = None
    mul_3081: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_78, sub_945)
    sum_375: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3081, [0, 2, 3]);  mul_3081 = None
    mul_3083: "f32[54]" = torch.ops.aten.mul.Tensor(sum_375, 1.814486863115111e-05)
    mul_3084: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_3085: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3083, mul_3084);  mul_3083 = mul_3084 = None
    unsqueeze_3042: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3085, 0);  mul_3085 = None
    unsqueeze_3043: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3042, 2);  unsqueeze_3042 = None
    unsqueeze_3044: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3043, 3);  unsqueeze_3043 = None
    mul_3086: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_75);  primals_75 = None
    unsqueeze_3045: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3086, 0);  mul_3086 = None
    unsqueeze_3046: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3045, 2);  unsqueeze_3045 = None
    unsqueeze_3047: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3046, 3);  unsqueeze_3046 = None
    mul_3087: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_945, unsqueeze_3044);  sub_945 = unsqueeze_3044 = None
    sub_947: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_78, mul_3087);  slice_78 = mul_3087 = None
    sub_948: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_947, unsqueeze_3029);  sub_947 = unsqueeze_3029 = None
    mul_3088: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_948, unsqueeze_3047);  sub_948 = unsqueeze_3047 = None
    mul_3089: "f32[54]" = torch.ops.aten.mul.Tensor(sum_375, squeeze_43);  sum_375 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_346 = torch.ops.aten.convolution_backward.default(mul_3088, convolution_25, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3088 = convolution_25 = primals_74 = None
    getitem_1524: "f32[8, 54, 83, 83]" = convolution_backward_346[0]
    getitem_1525: "f32[54, 54, 1, 1]" = convolution_backward_346[1];  convolution_backward_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_347 = torch.ops.aten.convolution_backward.default(getitem_1524, relu_12, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1524 = primals_73 = None
    getitem_1527: "f32[8, 54, 83, 83]" = convolution_backward_347[0]
    getitem_1528: "f32[54, 1, 3, 3]" = convolution_backward_347[1];  convolution_backward_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_187: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_187: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_187, full_default, getitem_1527);  le_187 = getitem_1527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_376: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_187, [0, 2, 3])
    sub_949: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_3050);  convolution_24 = unsqueeze_3050 = None
    mul_3090: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_187, sub_949)
    sum_377: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3090, [0, 2, 3]);  mul_3090 = None
    mul_3091: "f32[54]" = torch.ops.aten.mul.Tensor(sum_376, 1.814486863115111e-05)
    unsqueeze_3051: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3091, 0);  mul_3091 = None
    unsqueeze_3052: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3051, 2);  unsqueeze_3051 = None
    unsqueeze_3053: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3052, 3);  unsqueeze_3052 = None
    mul_3092: "f32[54]" = torch.ops.aten.mul.Tensor(sum_377, 1.814486863115111e-05)
    mul_3093: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_3094: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3092, mul_3093);  mul_3092 = mul_3093 = None
    unsqueeze_3054: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3094, 0);  mul_3094 = None
    unsqueeze_3055: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3054, 2);  unsqueeze_3054 = None
    unsqueeze_3056: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3055, 3);  unsqueeze_3055 = None
    mul_3095: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_71);  primals_71 = None
    unsqueeze_3057: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3095, 0);  mul_3095 = None
    unsqueeze_3058: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3057, 2);  unsqueeze_3057 = None
    unsqueeze_3059: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3058, 3);  unsqueeze_3058 = None
    mul_3096: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_949, unsqueeze_3056);  sub_949 = unsqueeze_3056 = None
    sub_951: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_187, mul_3096);  where_187 = mul_3096 = None
    sub_952: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_951, unsqueeze_3053);  sub_951 = unsqueeze_3053 = None
    mul_3097: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_952, unsqueeze_3059);  sub_952 = unsqueeze_3059 = None
    mul_3098: "f32[54]" = torch.ops.aten.mul.Tensor(sum_377, squeeze_40);  sum_377 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_348 = torch.ops.aten.convolution_backward.default(mul_3097, convolution_23, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3097 = convolution_23 = primals_70 = None
    getitem_1530: "f32[8, 96, 83, 83]" = convolution_backward_348[0]
    getitem_1531: "f32[54, 96, 1, 1]" = convolution_backward_348[1];  convolution_backward_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_349 = torch.ops.aten.convolution_backward.default(getitem_1530, constant_pad_nd_7, primals_7, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False]);  getitem_1530 = constant_pad_nd_7 = primals_7 = None
    getitem_1533: "f32[8, 96, 167, 167]" = convolution_backward_349[0]
    getitem_1534: "f32[96, 1, 3, 3]" = convolution_backward_349[1];  convolution_backward_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_72: "f32[8, 96, 165, 165]" = torch.ops.aten.constant_pad_nd.default(getitem_1533, [-1, -1, -1, -1]);  getitem_1533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_188: "f32[8, 96, 165, 165]" = torch.ops.aten.where.self(le_185, full_default, constant_pad_nd_72);  constant_pad_nd_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1195: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(where_185, where_188);  where_185 = where_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_39: "f32[8, 54, 167, 167]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_77, constant_pad_nd_3, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_73: "f32[8, 54, 165, 165]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_39, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1196: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(where_186, constant_pad_nd_73);  where_186 = constant_pad_nd_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_378: "f32[54]" = torch.ops.aten.sum.dim_IntList(slice_77, [0, 2, 3])
    sub_953: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_3062);  convolution_22 = unsqueeze_3062 = None
    mul_3099: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_77, sub_953)
    sum_379: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3099, [0, 2, 3]);  mul_3099 = None
    mul_3100: "f32[54]" = torch.ops.aten.mul.Tensor(sum_378, 1.814486863115111e-05)
    unsqueeze_3063: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3100, 0);  mul_3100 = None
    unsqueeze_3064: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3063, 2);  unsqueeze_3063 = None
    unsqueeze_3065: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3064, 3);  unsqueeze_3064 = None
    mul_3101: "f32[54]" = torch.ops.aten.mul.Tensor(sum_379, 1.814486863115111e-05)
    mul_3102: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_3103: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3101, mul_3102);  mul_3101 = mul_3102 = None
    unsqueeze_3066: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3103, 0);  mul_3103 = None
    unsqueeze_3067: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3066, 2);  unsqueeze_3066 = None
    unsqueeze_3068: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3067, 3);  unsqueeze_3067 = None
    mul_3104: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_68);  primals_68 = None
    unsqueeze_3069: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3104, 0);  mul_3104 = None
    unsqueeze_3070: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3069, 2);  unsqueeze_3069 = None
    unsqueeze_3071: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3070, 3);  unsqueeze_3070 = None
    mul_3105: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_953, unsqueeze_3068);  sub_953 = unsqueeze_3068 = None
    sub_955: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_77, mul_3105);  slice_77 = mul_3105 = None
    sub_956: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_955, unsqueeze_3065);  sub_955 = unsqueeze_3065 = None
    mul_3106: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_956, unsqueeze_3071);  sub_956 = unsqueeze_3071 = None
    mul_3107: "f32[54]" = torch.ops.aten.mul.Tensor(sum_379, squeeze_37);  sum_379 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_350 = torch.ops.aten.convolution_backward.default(mul_3106, convolution_21, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3106 = convolution_21 = primals_67 = None
    getitem_1536: "f32[8, 54, 83, 83]" = convolution_backward_350[0]
    getitem_1537: "f32[54, 54, 1, 1]" = convolution_backward_350[1];  convolution_backward_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_351 = torch.ops.aten.convolution_backward.default(getitem_1536, relu_10, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1536 = primals_66 = None
    getitem_1539: "f32[8, 54, 83, 83]" = convolution_backward_351[0]
    getitem_1540: "f32[54, 1, 3, 3]" = convolution_backward_351[1];  convolution_backward_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_189: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_189: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_189, full_default, getitem_1539);  le_189 = getitem_1539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_380: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_189, [0, 2, 3])
    sub_957: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_3074);  convolution_20 = unsqueeze_3074 = None
    mul_3108: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_189, sub_957)
    sum_381: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3108, [0, 2, 3]);  mul_3108 = None
    mul_3109: "f32[54]" = torch.ops.aten.mul.Tensor(sum_380, 1.814486863115111e-05)
    unsqueeze_3075: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3109, 0);  mul_3109 = None
    unsqueeze_3076: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3075, 2);  unsqueeze_3075 = None
    unsqueeze_3077: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3076, 3);  unsqueeze_3076 = None
    mul_3110: "f32[54]" = torch.ops.aten.mul.Tensor(sum_381, 1.814486863115111e-05)
    mul_3111: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_3112: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3110, mul_3111);  mul_3110 = mul_3111 = None
    unsqueeze_3078: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3112, 0);  mul_3112 = None
    unsqueeze_3079: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3078, 2);  unsqueeze_3078 = None
    unsqueeze_3080: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3079, 3);  unsqueeze_3079 = None
    mul_3113: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_64);  primals_64 = None
    unsqueeze_3081: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3113, 0);  mul_3113 = None
    unsqueeze_3082: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3081, 2);  unsqueeze_3081 = None
    unsqueeze_3083: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3082, 3);  unsqueeze_3082 = None
    mul_3114: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_957, unsqueeze_3080);  sub_957 = unsqueeze_3080 = None
    sub_959: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_189, mul_3114);  where_189 = mul_3114 = None
    sub_960: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_959, unsqueeze_3077);  sub_959 = unsqueeze_3077 = None
    mul_3115: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_960, unsqueeze_3083);  sub_960 = unsqueeze_3083 = None
    mul_3116: "f32[54]" = torch.ops.aten.mul.Tensor(sum_381, squeeze_34);  sum_381 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_352 = torch.ops.aten.convolution_backward.default(mul_3115, convolution_19, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3115 = convolution_19 = primals_63 = None
    getitem_1542: "f32[8, 54, 83, 83]" = convolution_backward_352[0]
    getitem_1543: "f32[54, 54, 1, 1]" = convolution_backward_352[1];  convolution_backward_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_353 = torch.ops.aten.convolution_backward.default(getitem_1542, relu_9, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1542 = primals_62 = None
    getitem_1545: "f32[8, 54, 83, 83]" = convolution_backward_353[0]
    getitem_1546: "f32[54, 1, 3, 3]" = convolution_backward_353[1];  convolution_backward_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    le_190: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_190: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_190, full_default, getitem_1545);  le_190 = getitem_1545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1197: "f32[8, 54, 83, 83]" = torch.ops.aten.add.Tensor(slice_76, where_190);  slice_76 = where_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_382: "f32[54]" = torch.ops.aten.sum.dim_IntList(add_1197, [0, 2, 3])
    sub_961: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_3086);  convolution_18 = unsqueeze_3086 = None
    mul_3117: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(add_1197, sub_961)
    sum_383: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3117, [0, 2, 3]);  mul_3117 = None
    mul_3118: "f32[54]" = torch.ops.aten.mul.Tensor(sum_382, 1.814486863115111e-05)
    unsqueeze_3087: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3118, 0);  mul_3118 = None
    unsqueeze_3088: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3087, 2);  unsqueeze_3087 = None
    unsqueeze_3089: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3088, 3);  unsqueeze_3088 = None
    mul_3119: "f32[54]" = torch.ops.aten.mul.Tensor(sum_383, 1.814486863115111e-05)
    mul_3120: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_3121: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3119, mul_3120);  mul_3119 = mul_3120 = None
    unsqueeze_3090: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3121, 0);  mul_3121 = None
    unsqueeze_3091: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3090, 2);  unsqueeze_3090 = None
    unsqueeze_3092: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3091, 3);  unsqueeze_3091 = None
    mul_3122: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_60);  primals_60 = None
    unsqueeze_3093: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3122, 0);  mul_3122 = None
    unsqueeze_3094: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3093, 2);  unsqueeze_3093 = None
    unsqueeze_3095: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3094, 3);  unsqueeze_3094 = None
    mul_3123: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_961, unsqueeze_3092);  sub_961 = unsqueeze_3092 = None
    sub_963: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(add_1197, mul_3123);  mul_3123 = None
    sub_964: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_963, unsqueeze_3089);  sub_963 = None
    mul_3124: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_964, unsqueeze_3095);  sub_964 = unsqueeze_3095 = None
    mul_3125: "f32[54]" = torch.ops.aten.mul.Tensor(sum_383, squeeze_31);  sum_383 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_354 = torch.ops.aten.convolution_backward.default(mul_3124, convolution_17, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3124 = convolution_17 = primals_59 = None
    getitem_1548: "f32[8, 54, 83, 83]" = convolution_backward_354[0]
    getitem_1549: "f32[54, 54, 1, 1]" = convolution_backward_354[1];  convolution_backward_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_355 = torch.ops.aten.convolution_backward.default(getitem_1548, relu_8, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1548 = primals_58 = None
    getitem_1551: "f32[8, 54, 83, 83]" = convolution_backward_355[0]
    getitem_1552: "f32[54, 1, 3, 3]" = convolution_backward_355[1];  convolution_backward_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_191: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_191: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_191, full_default, getitem_1551);  le_191 = getitem_1551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_384: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_191, [0, 2, 3])
    sub_965: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_3098);  convolution_16 = unsqueeze_3098 = None
    mul_3126: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_191, sub_965)
    sum_385: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3126, [0, 2, 3]);  mul_3126 = None
    mul_3127: "f32[54]" = torch.ops.aten.mul.Tensor(sum_384, 1.814486863115111e-05)
    unsqueeze_3099: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3127, 0);  mul_3127 = None
    unsqueeze_3100: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3099, 2);  unsqueeze_3099 = None
    unsqueeze_3101: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3100, 3);  unsqueeze_3100 = None
    mul_3128: "f32[54]" = torch.ops.aten.mul.Tensor(sum_385, 1.814486863115111e-05)
    mul_3129: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_3130: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3128, mul_3129);  mul_3128 = mul_3129 = None
    unsqueeze_3102: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3130, 0);  mul_3130 = None
    unsqueeze_3103: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3102, 2);  unsqueeze_3102 = None
    unsqueeze_3104: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3103, 3);  unsqueeze_3103 = None
    mul_3131: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_56);  primals_56 = None
    unsqueeze_3105: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3131, 0);  mul_3131 = None
    unsqueeze_3106: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3105, 2);  unsqueeze_3105 = None
    unsqueeze_3107: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3106, 3);  unsqueeze_3106 = None
    mul_3132: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_965, unsqueeze_3104);  sub_965 = unsqueeze_3104 = None
    sub_967: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_191, mul_3132);  where_191 = mul_3132 = None
    sub_968: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_967, unsqueeze_3101);  sub_967 = unsqueeze_3101 = None
    mul_3133: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_968, unsqueeze_3107);  sub_968 = unsqueeze_3107 = None
    mul_3134: "f32[54]" = torch.ops.aten.mul.Tensor(sum_385, squeeze_28);  sum_385 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_356 = torch.ops.aten.convolution_backward.default(mul_3133, convolution_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3133 = convolution_15 = primals_55 = None
    getitem_1554: "f32[8, 54, 83, 83]" = convolution_backward_356[0]
    getitem_1555: "f32[54, 54, 1, 1]" = convolution_backward_356[1];  convolution_backward_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_357 = torch.ops.aten.convolution_backward.default(getitem_1554, constant_pad_nd_5, primals_6, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1554 = constant_pad_nd_5 = primals_6 = None
    getitem_1557: "f32[8, 54, 167, 167]" = convolution_backward_357[0]
    getitem_1558: "f32[54, 1, 3, 3]" = convolution_backward_357[1];  convolution_backward_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_74: "f32[8, 54, 165, 165]" = torch.ops.aten.constant_pad_nd.default(getitem_1557, [-1, -1, -1, -1]);  getitem_1557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_192: "f32[8, 54, 165, 165]" = torch.ops.aten.where.self(le_186, full_default, constant_pad_nd_74);  constant_pad_nd_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1198: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(add_1196, where_192);  add_1196 = where_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_969: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_3110);  convolution_14 = unsqueeze_3110 = None
    mul_3135: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(add_1197, sub_969)
    sum_387: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3135, [0, 2, 3]);  mul_3135 = None
    mul_3137: "f32[54]" = torch.ops.aten.mul.Tensor(sum_387, 1.814486863115111e-05)
    mul_3138: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_3139: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3137, mul_3138);  mul_3137 = mul_3138 = None
    unsqueeze_3114: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3139, 0);  mul_3139 = None
    unsqueeze_3115: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3114, 2);  unsqueeze_3114 = None
    unsqueeze_3116: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3115, 3);  unsqueeze_3115 = None
    mul_3140: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_53);  primals_53 = None
    unsqueeze_3117: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3140, 0);  mul_3140 = None
    unsqueeze_3118: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3117, 2);  unsqueeze_3117 = None
    unsqueeze_3119: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3118, 3);  unsqueeze_3118 = None
    mul_3141: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_969, unsqueeze_3116);  sub_969 = unsqueeze_3116 = None
    sub_971: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(add_1197, mul_3141);  add_1197 = mul_3141 = None
    sub_972: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_971, unsqueeze_3089);  sub_971 = unsqueeze_3089 = None
    mul_3142: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_972, unsqueeze_3119);  sub_972 = unsqueeze_3119 = None
    mul_3143: "f32[54]" = torch.ops.aten.mul.Tensor(sum_387, squeeze_25);  sum_387 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_358 = torch.ops.aten.convolution_backward.default(mul_3142, convolution_13, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3142 = convolution_13 = primals_52 = None
    getitem_1560: "f32[8, 54, 83, 83]" = convolution_backward_358[0]
    getitem_1561: "f32[54, 54, 1, 1]" = convolution_backward_358[1];  convolution_backward_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_359 = torch.ops.aten.convolution_backward.default(getitem_1560, relu_6, primals_51, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1560 = primals_51 = None
    getitem_1563: "f32[8, 54, 83, 83]" = convolution_backward_359[0]
    getitem_1564: "f32[54, 1, 5, 5]" = convolution_backward_359[1];  convolution_backward_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_193: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_193: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_193, full_default, getitem_1563);  le_193 = getitem_1563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_388: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_193, [0, 2, 3])
    sub_973: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_3122);  convolution_12 = unsqueeze_3122 = None
    mul_3144: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_193, sub_973)
    sum_389: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3144, [0, 2, 3]);  mul_3144 = None
    mul_3145: "f32[54]" = torch.ops.aten.mul.Tensor(sum_388, 1.814486863115111e-05)
    unsqueeze_3123: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3145, 0);  mul_3145 = None
    unsqueeze_3124: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3123, 2);  unsqueeze_3123 = None
    unsqueeze_3125: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3124, 3);  unsqueeze_3124 = None
    mul_3146: "f32[54]" = torch.ops.aten.mul.Tensor(sum_389, 1.814486863115111e-05)
    mul_3147: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_3148: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3146, mul_3147);  mul_3146 = mul_3147 = None
    unsqueeze_3126: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3148, 0);  mul_3148 = None
    unsqueeze_3127: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3126, 2);  unsqueeze_3126 = None
    unsqueeze_3128: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3127, 3);  unsqueeze_3127 = None
    mul_3149: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_49);  primals_49 = None
    unsqueeze_3129: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3149, 0);  mul_3149 = None
    unsqueeze_3130: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3129, 2);  unsqueeze_3129 = None
    unsqueeze_3131: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3130, 3);  unsqueeze_3130 = None
    mul_3150: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_973, unsqueeze_3128);  sub_973 = unsqueeze_3128 = None
    sub_975: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_193, mul_3150);  where_193 = mul_3150 = None
    sub_976: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_975, unsqueeze_3125);  sub_975 = unsqueeze_3125 = None
    mul_3151: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_976, unsqueeze_3131);  sub_976 = unsqueeze_3131 = None
    mul_3152: "f32[54]" = torch.ops.aten.mul.Tensor(sum_389, squeeze_22);  sum_389 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_360 = torch.ops.aten.convolution_backward.default(mul_3151, convolution_11, primals_48, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3151 = convolution_11 = primals_48 = None
    getitem_1566: "f32[8, 54, 83, 83]" = convolution_backward_360[0]
    getitem_1567: "f32[54, 54, 1, 1]" = convolution_backward_360[1];  convolution_backward_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_361 = torch.ops.aten.convolution_backward.default(getitem_1566, constant_pad_nd_4, primals_5, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1566 = constant_pad_nd_4 = primals_5 = None
    getitem_1569: "f32[8, 54, 169, 169]" = convolution_backward_361[0]
    getitem_1570: "f32[54, 1, 5, 5]" = convolution_backward_361[1];  convolution_backward_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_75: "f32[8, 54, 165, 165]" = torch.ops.aten.constant_pad_nd.default(getitem_1569, [-2, -2, -2, -2]);  getitem_1569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_194: "f32[8, 54, 165, 165]" = torch.ops.aten.where.self(le_186, full_default, constant_pad_nd_75);  constant_pad_nd_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1199: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(add_1198, where_194);  add_1198 = where_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_40: "f32[8, 54, 167, 167]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_75, constant_pad_nd_3, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_17);  constant_pad_nd_3 = getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_76: "f32[8, 54, 165, 165]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_40, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1200: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(add_1199, constant_pad_nd_76);  add_1199 = constant_pad_nd_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sum_390: "f32[54]" = torch.ops.aten.sum.dim_IntList(slice_75, [0, 2, 3])
    sub_977: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_3134);  convolution_10 = unsqueeze_3134 = None
    mul_3153: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_75, sub_977)
    sum_391: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3153, [0, 2, 3]);  mul_3153 = None
    mul_3154: "f32[54]" = torch.ops.aten.mul.Tensor(sum_390, 1.814486863115111e-05)
    unsqueeze_3135: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3154, 0);  mul_3154 = None
    unsqueeze_3136: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3135, 2);  unsqueeze_3135 = None
    unsqueeze_3137: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3136, 3);  unsqueeze_3136 = None
    mul_3155: "f32[54]" = torch.ops.aten.mul.Tensor(sum_391, 1.814486863115111e-05)
    mul_3156: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_3157: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3155, mul_3156);  mul_3155 = mul_3156 = None
    unsqueeze_3138: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3157, 0);  mul_3157 = None
    unsqueeze_3139: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3138, 2);  unsqueeze_3138 = None
    unsqueeze_3140: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3139, 3);  unsqueeze_3139 = None
    mul_3158: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_46);  primals_46 = None
    unsqueeze_3141: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3158, 0);  mul_3158 = None
    unsqueeze_3142: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3141, 2);  unsqueeze_3141 = None
    unsqueeze_3143: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3142, 3);  unsqueeze_3142 = None
    mul_3159: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_977, unsqueeze_3140);  sub_977 = unsqueeze_3140 = None
    sub_979: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_75, mul_3159);  slice_75 = mul_3159 = None
    sub_980: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_979, unsqueeze_3137);  sub_979 = unsqueeze_3137 = None
    mul_3160: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_980, unsqueeze_3143);  sub_980 = unsqueeze_3143 = None
    mul_3161: "f32[54]" = torch.ops.aten.mul.Tensor(sum_391, squeeze_19);  sum_391 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_362 = torch.ops.aten.convolution_backward.default(mul_3160, convolution_9, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3160 = convolution_9 = primals_45 = None
    getitem_1572: "f32[8, 54, 83, 83]" = convolution_backward_362[0]
    getitem_1573: "f32[54, 54, 1, 1]" = convolution_backward_362[1];  convolution_backward_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_363 = torch.ops.aten.convolution_backward.default(getitem_1572, relu_4, primals_44, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1572 = primals_44 = None
    getitem_1575: "f32[8, 54, 83, 83]" = convolution_backward_363[0]
    getitem_1576: "f32[54, 1, 7, 7]" = convolution_backward_363[1];  convolution_backward_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_195: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_195: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_195, full_default, getitem_1575);  le_195 = getitem_1575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_392: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_195, [0, 2, 3])
    sub_981: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_3146);  convolution_8 = unsqueeze_3146 = None
    mul_3162: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_195, sub_981)
    sum_393: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3162, [0, 2, 3]);  mul_3162 = None
    mul_3163: "f32[54]" = torch.ops.aten.mul.Tensor(sum_392, 1.814486863115111e-05)
    unsqueeze_3147: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3163, 0);  mul_3163 = None
    unsqueeze_3148: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3147, 2);  unsqueeze_3147 = None
    unsqueeze_3149: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3148, 3);  unsqueeze_3148 = None
    mul_3164: "f32[54]" = torch.ops.aten.mul.Tensor(sum_393, 1.814486863115111e-05)
    mul_3165: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_3166: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3164, mul_3165);  mul_3164 = mul_3165 = None
    unsqueeze_3150: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3166, 0);  mul_3166 = None
    unsqueeze_3151: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3150, 2);  unsqueeze_3150 = None
    unsqueeze_3152: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3151, 3);  unsqueeze_3151 = None
    mul_3167: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_42);  primals_42 = None
    unsqueeze_3153: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3167, 0);  mul_3167 = None
    unsqueeze_3154: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3153, 2);  unsqueeze_3153 = None
    unsqueeze_3155: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3154, 3);  unsqueeze_3154 = None
    mul_3168: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_981, unsqueeze_3152);  sub_981 = unsqueeze_3152 = None
    sub_983: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_195, mul_3168);  where_195 = mul_3168 = None
    sub_984: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_983, unsqueeze_3149);  sub_983 = unsqueeze_3149 = None
    mul_3169: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_984, unsqueeze_3155);  sub_984 = unsqueeze_3155 = None
    mul_3170: "f32[54]" = torch.ops.aten.mul.Tensor(sum_393, squeeze_16);  sum_393 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_364 = torch.ops.aten.convolution_backward.default(mul_3169, convolution_7, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3169 = convolution_7 = primals_41 = None
    getitem_1578: "f32[8, 54, 83, 83]" = convolution_backward_364[0]
    getitem_1579: "f32[54, 54, 1, 1]" = convolution_backward_364[1];  convolution_backward_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_365 = torch.ops.aten.convolution_backward.default(getitem_1578, constant_pad_nd_2, primals_4, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1578 = constant_pad_nd_2 = primals_4 = None
    getitem_1581: "f32[8, 54, 171, 171]" = convolution_backward_365[0]
    getitem_1582: "f32[54, 1, 7, 7]" = convolution_backward_365[1];  convolution_backward_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_77: "f32[8, 54, 165, 165]" = torch.ops.aten.constant_pad_nd.default(getitem_1581, [-3, -3, -3, -3]);  getitem_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_196: "f32[8, 54, 165, 165]" = torch.ops.aten.where.self(le_186, full_default, constant_pad_nd_77);  le_186 = constant_pad_nd_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1201: "f32[8, 54, 165, 165]" = torch.ops.aten.add.Tensor(add_1200, where_196);  add_1200 = where_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:106, code: x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    sum_394: "f32[54]" = torch.ops.aten.sum.dim_IntList(slice_74, [0, 2, 3])
    sub_985: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_3158);  convolution_6 = unsqueeze_3158 = None
    mul_3171: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_74, sub_985)
    sum_395: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3171, [0, 2, 3]);  mul_3171 = None
    mul_3172: "f32[54]" = torch.ops.aten.mul.Tensor(sum_394, 1.814486863115111e-05)
    unsqueeze_3159: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3172, 0);  mul_3172 = None
    unsqueeze_3160: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3159, 2);  unsqueeze_3159 = None
    unsqueeze_3161: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3160, 3);  unsqueeze_3160 = None
    mul_3173: "f32[54]" = torch.ops.aten.mul.Tensor(sum_395, 1.814486863115111e-05)
    mul_3174: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_3175: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3173, mul_3174);  mul_3173 = mul_3174 = None
    unsqueeze_3162: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3175, 0);  mul_3175 = None
    unsqueeze_3163: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3162, 2);  unsqueeze_3162 = None
    unsqueeze_3164: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3163, 3);  unsqueeze_3163 = None
    mul_3176: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_39);  primals_39 = None
    unsqueeze_3165: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3176, 0);  mul_3176 = None
    unsqueeze_3166: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3165, 2);  unsqueeze_3165 = None
    unsqueeze_3167: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3166, 3);  unsqueeze_3166 = None
    mul_3177: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_985, unsqueeze_3164);  sub_985 = unsqueeze_3164 = None
    sub_987: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_74, mul_3177);  mul_3177 = None
    sub_988: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_987, unsqueeze_3161);  sub_987 = None
    mul_3178: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_988, unsqueeze_3167);  sub_988 = unsqueeze_3167 = None
    mul_3179: "f32[54]" = torch.ops.aten.mul.Tensor(sum_395, squeeze_13);  sum_395 = squeeze_13 = None
    convolution_backward_366 = torch.ops.aten.convolution_backward.default(mul_3178, getitem_8, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3178 = getitem_8 = primals_38 = None
    getitem_1584: "f32[8, 96, 83, 83]" = convolution_backward_366[0]
    getitem_1585: "f32[54, 96, 1, 1]" = convolution_backward_366[1];  convolution_backward_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/pool2d_same.py:53, code: return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    max_pool2d_with_indices_backward_41: "f32[8, 96, 167, 167]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_1584, constant_pad_nd_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_9);  getitem_1584 = constant_pad_nd_1 = getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_78: "f32[8, 96, 165, 165]" = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_41, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    add_1202: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(add_1195, constant_pad_nd_78);  add_1195 = constant_pad_nd_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:58, code: x = self.bn_sep_2(x)
    sub_989: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_3170);  convolution_5 = unsqueeze_3170 = None
    mul_3180: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(slice_74, sub_989)
    sum_397: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3180, [0, 2, 3]);  mul_3180 = None
    mul_3182: "f32[54]" = torch.ops.aten.mul.Tensor(sum_397, 1.814486863115111e-05)
    mul_3183: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_3184: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3182, mul_3183);  mul_3182 = mul_3183 = None
    unsqueeze_3174: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3184, 0);  mul_3184 = None
    unsqueeze_3175: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3174, 2);  unsqueeze_3174 = None
    unsqueeze_3176: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3175, 3);  unsqueeze_3175 = None
    mul_3185: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_36);  primals_36 = None
    unsqueeze_3177: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3185, 0);  mul_3185 = None
    unsqueeze_3178: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3177, 2);  unsqueeze_3177 = None
    unsqueeze_3179: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3178, 3);  unsqueeze_3178 = None
    mul_3186: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_989, unsqueeze_3176);  sub_989 = unsqueeze_3176 = None
    sub_991: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(slice_74, mul_3186);  slice_74 = mul_3186 = None
    sub_992: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_991, unsqueeze_3161);  sub_991 = unsqueeze_3161 = None
    mul_3187: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_992, unsqueeze_3179);  sub_992 = unsqueeze_3179 = None
    mul_3188: "f32[54]" = torch.ops.aten.mul.Tensor(sum_397, squeeze_10);  sum_397 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_367 = torch.ops.aten.convolution_backward.default(mul_3187, convolution_4, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3187 = convolution_4 = primals_35 = None
    getitem_1587: "f32[8, 54, 83, 83]" = convolution_backward_367[0]
    getitem_1588: "f32[54, 54, 1, 1]" = convolution_backward_367[1];  convolution_backward_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:33, code: x = self.depthwise_conv2d(x)
    convolution_backward_368 = torch.ops.aten.convolution_backward.default(getitem_1587, relu_2, primals_34, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 54, [True, True, False]);  getitem_1587 = primals_34 = None
    getitem_1590: "f32[8, 54, 83, 83]" = convolution_backward_368[0]
    getitem_1591: "f32[54, 1, 5, 5]" = convolution_backward_368[1];  convolution_backward_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:56, code: x = self.act_2(x)
    le_197: "b8[8, 54, 83, 83]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_197: "f32[8, 54, 83, 83]" = torch.ops.aten.where.self(le_197, full_default, getitem_1590);  le_197 = getitem_1590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:55, code: x = self.bn_sep_1(x)
    sum_398: "f32[54]" = torch.ops.aten.sum.dim_IntList(where_197, [0, 2, 3])
    sub_993: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_3182);  convolution_3 = unsqueeze_3182 = None
    mul_3189: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(where_197, sub_993)
    sum_399: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3189, [0, 2, 3]);  mul_3189 = None
    mul_3190: "f32[54]" = torch.ops.aten.mul.Tensor(sum_398, 1.814486863115111e-05)
    unsqueeze_3183: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3190, 0);  mul_3190 = None
    unsqueeze_3184: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3183, 2);  unsqueeze_3183 = None
    unsqueeze_3185: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3184, 3);  unsqueeze_3184 = None
    mul_3191: "f32[54]" = torch.ops.aten.mul.Tensor(sum_399, 1.814486863115111e-05)
    mul_3192: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_3193: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3191, mul_3192);  mul_3191 = mul_3192 = None
    unsqueeze_3186: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3193, 0);  mul_3193 = None
    unsqueeze_3187: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3186, 2);  unsqueeze_3186 = None
    unsqueeze_3188: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3187, 3);  unsqueeze_3187 = None
    mul_3194: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_32);  primals_32 = None
    unsqueeze_3189: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3194, 0);  mul_3194 = None
    unsqueeze_3190: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3189, 2);  unsqueeze_3189 = None
    unsqueeze_3191: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3190, 3);  unsqueeze_3190 = None
    mul_3195: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_993, unsqueeze_3188);  sub_993 = unsqueeze_3188 = None
    sub_995: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(where_197, mul_3195);  where_197 = mul_3195 = None
    sub_996: "f32[8, 54, 83, 83]" = torch.ops.aten.sub.Tensor(sub_995, unsqueeze_3185);  sub_995 = unsqueeze_3185 = None
    mul_3196: "f32[8, 54, 83, 83]" = torch.ops.aten.mul.Tensor(sub_996, unsqueeze_3191);  sub_996 = unsqueeze_3191 = None
    mul_3197: "f32[54]" = torch.ops.aten.mul.Tensor(sum_399, squeeze_7);  sum_399 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:34, code: x = self.pointwise_conv2d(x)
    convolution_backward_369 = torch.ops.aten.convolution_backward.default(mul_3196, convolution_2, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3196 = convolution_2 = primals_31 = None
    getitem_1593: "f32[8, 96, 83, 83]" = convolution_backward_369[0]
    getitem_1594: "f32[54, 96, 1, 1]" = convolution_backward_369[1];  convolution_backward_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_370 = torch.ops.aten.convolution_backward.default(getitem_1593, constant_pad_nd, primals_3, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False]);  getitem_1593 = constant_pad_nd = primals_3 = None
    getitem_1596: "f32[8, 96, 169, 169]" = convolution_backward_370[0]
    getitem_1597: "f32[96, 1, 5, 5]" = convolution_backward_370[1];  convolution_backward_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_79: "f32[8, 96, 165, 165]" = torch.ops.aten.constant_pad_nd.default(getitem_1596, [-2, -2, -2, -2]);  getitem_1596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    where_198: "f32[8, 96, 165, 165]" = torch.ops.aten.where.self(le_185, full_default, constant_pad_nd_79);  constant_pad_nd_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:53, code: x = self.act_1(x)
    add_1203: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(add_1202, where_198);  add_1202 = where_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:74, code: x = self.bn(x)
    sum_400: "f32[54]" = torch.ops.aten.sum.dim_IntList(add_1201, [0, 2, 3])
    sub_997: "f32[8, 54, 165, 165]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_3194);  convolution_1 = unsqueeze_3194 = None
    mul_3198: "f32[8, 54, 165, 165]" = torch.ops.aten.mul.Tensor(add_1201, sub_997)
    sum_401: "f32[54]" = torch.ops.aten.sum.dim_IntList(mul_3198, [0, 2, 3]);  mul_3198 = None
    mul_3199: "f32[54]" = torch.ops.aten.mul.Tensor(sum_400, 4.591368227731864e-06)
    unsqueeze_3195: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3199, 0);  mul_3199 = None
    unsqueeze_3196: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3195, 2);  unsqueeze_3195 = None
    unsqueeze_3197: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3196, 3);  unsqueeze_3196 = None
    mul_3200: "f32[54]" = torch.ops.aten.mul.Tensor(sum_401, 4.591368227731864e-06)
    mul_3201: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_3202: "f32[54]" = torch.ops.aten.mul.Tensor(mul_3200, mul_3201);  mul_3200 = mul_3201 = None
    unsqueeze_3198: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3202, 0);  mul_3202 = None
    unsqueeze_3199: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3198, 2);  unsqueeze_3198 = None
    unsqueeze_3200: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3199, 3);  unsqueeze_3199 = None
    mul_3203: "f32[54]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_29);  primals_29 = None
    unsqueeze_3201: "f32[1, 54]" = torch.ops.aten.unsqueeze.default(mul_3203, 0);  mul_3203 = None
    unsqueeze_3202: "f32[1, 54, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3201, 2);  unsqueeze_3201 = None
    unsqueeze_3203: "f32[1, 54, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3202, 3);  unsqueeze_3202 = None
    mul_3204: "f32[8, 54, 165, 165]" = torch.ops.aten.mul.Tensor(sub_997, unsqueeze_3200);  sub_997 = unsqueeze_3200 = None
    sub_999: "f32[8, 54, 165, 165]" = torch.ops.aten.sub.Tensor(add_1201, mul_3204);  add_1201 = mul_3204 = None
    sub_1000: "f32[8, 54, 165, 165]" = torch.ops.aten.sub.Tensor(sub_999, unsqueeze_3197);  sub_999 = unsqueeze_3197 = None
    mul_3205: "f32[8, 54, 165, 165]" = torch.ops.aten.mul.Tensor(sub_1000, unsqueeze_3203);  sub_1000 = unsqueeze_3203 = None
    mul_3206: "f32[54]" = torch.ops.aten.mul.Tensor(sum_401, squeeze_4);  sum_401 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:73, code: x = self.conv(x)
    convolution_backward_371 = torch.ops.aten.convolution_backward.default(mul_3205, relu, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_3205 = relu = primals_28 = None
    getitem_1599: "f32[8, 96, 165, 165]" = convolution_backward_371[0]
    getitem_1600: "f32[54, 96, 1, 1]" = convolution_backward_371[1];  convolution_backward_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    where_199: "f32[8, 96, 165, 165]" = torch.ops.aten.where.self(le_185, full_default, getitem_1599);  le_185 = full_default = getitem_1599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pnasnet.py:72, code: x = self.act(x)
    add_1204: "f32[8, 96, 165, 165]" = torch.ops.aten.add.Tensor(add_1203, where_199);  add_1203 = where_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_402: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_1204, [0, 2, 3])
    sub_1001: "f32[8, 96, 165, 165]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_3206);  convolution = unsqueeze_3206 = None
    mul_3207: "f32[8, 96, 165, 165]" = torch.ops.aten.mul.Tensor(add_1204, sub_1001)
    sum_403: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_3207, [0, 2, 3]);  mul_3207 = None
    mul_3208: "f32[96]" = torch.ops.aten.mul.Tensor(sum_402, 4.591368227731864e-06)
    unsqueeze_3207: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_3208, 0);  mul_3208 = None
    unsqueeze_3208: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3207, 2);  unsqueeze_3207 = None
    unsqueeze_3209: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3208, 3);  unsqueeze_3208 = None
    mul_3209: "f32[96]" = torch.ops.aten.mul.Tensor(sum_403, 4.591368227731864e-06)
    mul_3210: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_3211: "f32[96]" = torch.ops.aten.mul.Tensor(mul_3209, mul_3210);  mul_3209 = mul_3210 = None
    unsqueeze_3210: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_3211, 0);  mul_3211 = None
    unsqueeze_3211: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3210, 2);  unsqueeze_3210 = None
    unsqueeze_3212: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3211, 3);  unsqueeze_3211 = None
    mul_3212: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_3213: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_3212, 0);  mul_3212 = None
    unsqueeze_3214: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3213, 2);  unsqueeze_3213 = None
    unsqueeze_3215: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3214, 3);  unsqueeze_3214 = None
    mul_3213: "f32[8, 96, 165, 165]" = torch.ops.aten.mul.Tensor(sub_1001, unsqueeze_3212);  sub_1001 = unsqueeze_3212 = None
    sub_1003: "f32[8, 96, 165, 165]" = torch.ops.aten.sub.Tensor(add_1204, mul_3213);  add_1204 = mul_3213 = None
    sub_1004: "f32[8, 96, 165, 165]" = torch.ops.aten.sub.Tensor(sub_1003, unsqueeze_3209);  sub_1003 = unsqueeze_3209 = None
    mul_3214: "f32[8, 96, 165, 165]" = torch.ops.aten.mul.Tensor(sub_1004, unsqueeze_3215);  sub_1004 = unsqueeze_3215 = None
    mul_3215: "f32[96]" = torch.ops.aten.mul.Tensor(sum_403, squeeze_1);  sum_403 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_372 = torch.ops.aten.convolution_backward.default(mul_3214, primals_1381, primals_27, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_3214 = primals_1381 = primals_27 = None
    getitem_1603: "f32[96, 3, 3, 3]" = convolution_backward_372[1];  convolution_backward_372 = None
    return [mul_3215, sum_402, getitem_1597, getitem_1582, getitem_1570, getitem_1558, getitem_1534, getitem_1522, getitem_1510, getitem_1498, getitem_1486, getitem_1474, getitem_1450, getitem_1438, getitem_1114, getitem_1102, getitem_1090, getitem_1078, getitem_1054, getitem_1042, getitem_796, getitem_784, getitem_772, getitem_760, getitem_736, getitem_724, getitem_1603, getitem_1600, mul_3206, sum_400, getitem_1594, mul_3197, sum_398, getitem_1591, getitem_1588, mul_3188, sum_394, getitem_1585, mul_3179, sum_394, getitem_1579, mul_3170, sum_392, getitem_1576, getitem_1573, mul_3161, sum_390, getitem_1567, mul_3152, sum_388, getitem_1564, getitem_1561, mul_3143, sum_382, getitem_1555, mul_3134, sum_384, getitem_1552, getitem_1549, mul_3125, sum_382, getitem_1546, getitem_1543, mul_3116, sum_380, getitem_1540, getitem_1537, mul_3107, sum_378, getitem_1531, mul_3098, sum_376, getitem_1528, getitem_1525, mul_3089, sum_372, mul_3080, sum_372, getitem_1519, getitem_1516, mul_3071, sum_370, getitem_1513, mul_3062, sum_368, getitem_1507, mul_3053, sum_366, getitem_1504, getitem_1501, mul_3044, sum_364, getitem_1495, mul_3035, sum_362, getitem_1492, getitem_1489, mul_3026, sum_360, getitem_1483, mul_3017, sum_358, getitem_1480, getitem_1477, mul_3008, sum_352, getitem_1471, mul_2999, sum_354, getitem_1468, getitem_1465, mul_2990, sum_352, getitem_1462, getitem_1459, mul_2981, sum_350, getitem_1456, getitem_1453, mul_2972, sum_348, getitem_1447, mul_2963, sum_346, getitem_1444, getitem_1441, mul_2954, sum_342, mul_2945, sum_342, getitem_1435, getitem_1432, mul_2936, sum_340, getitem_1429, mul_2927, sum_338, getitem_1426, getitem_1423, mul_2918, sum_336, getitem_1420, getitem_1417, mul_2909, sum_334, getitem_1414, getitem_1411, mul_2900, sum_332, getitem_1408, getitem_1405, mul_2891, sum_330, getitem_1402, getitem_1399, mul_2882, sum_328, getitem_1396, getitem_1393, mul_2873, sum_322, getitem_1390, getitem_1387, mul_2864, sum_324, getitem_1384, getitem_1381, mul_2855, sum_322, getitem_1378, getitem_1375, mul_2846, sum_320, getitem_1372, getitem_1369, mul_2837, sum_318, getitem_1366, getitem_1363, mul_2828, sum_316, getitem_1360, getitem_1357, mul_2819, sum_314, getitem_1354, mul_2810, sum_312, getitem_1351, mul_2801, sum_310, getitem_1348, getitem_1345, mul_2792, sum_308, getitem_1342, getitem_1339, mul_2783, sum_306, getitem_1336, getitem_1333, mul_2774, sum_304, getitem_1330, getitem_1327, mul_2765, sum_302, getitem_1324, getitem_1321, mul_2756, sum_300, getitem_1318, getitem_1315, mul_2747, sum_294, getitem_1312, getitem_1309, mul_2738, sum_296, getitem_1306, getitem_1303, mul_2729, sum_294, getitem_1300, getitem_1297, mul_2720, sum_292, getitem_1294, getitem_1291, mul_2711, sum_290, getitem_1288, getitem_1285, mul_2702, sum_288, getitem_1282, getitem_1279, mul_2693, sum_286, getitem_1276, mul_2684, sum_284, getitem_1273, mul_2675, sum_282, getitem_1270, getitem_1267, mul_2666, sum_280, getitem_1264, getitem_1261, mul_2657, sum_278, getitem_1258, getitem_1255, mul_2648, sum_276, getitem_1252, getitem_1249, mul_2639, sum_274, getitem_1246, getitem_1243, mul_2630, sum_272, getitem_1240, getitem_1237, mul_2621, sum_266, getitem_1234, getitem_1231, mul_2612, sum_268, getitem_1228, getitem_1225, mul_2603, sum_266, getitem_1222, getitem_1219, mul_2594, sum_264, getitem_1216, getitem_1213, mul_2585, sum_262, getitem_1210, getitem_1207, mul_2576, sum_260, getitem_1204, getitem_1201, mul_2567, sum_258, getitem_1198, mul_2558, sum_256, getitem_1195, mul_2549, sum_254, getitem_1192, getitem_1189, mul_2540, sum_252, getitem_1186, getitem_1183, mul_2531, sum_250, getitem_1180, getitem_1177, mul_2522, sum_248, getitem_1174, getitem_1171, mul_2513, sum_246, getitem_1168, getitem_1165, mul_2504, sum_244, getitem_1162, getitem_1159, mul_2495, sum_238, getitem_1156, getitem_1153, mul_2486, sum_240, getitem_1150, getitem_1147, mul_2477, sum_238, getitem_1144, getitem_1141, mul_2468, sum_236, getitem_1138, getitem_1135, mul_2459, sum_234, getitem_1132, getitem_1129, mul_2450, sum_232, getitem_1126, getitem_1123, mul_2441, sum_230, getitem_1120, mul_2432, sum_228, getitem_1117, mul_2423, sum_226, getitem_1111, mul_2414, sum_224, getitem_1108, getitem_1105, mul_2405, sum_222, getitem_1099, mul_2396, sum_220, getitem_1096, getitem_1093, mul_2387, sum_218, getitem_1087, mul_2378, sum_216, getitem_1084, getitem_1081, mul_2369, sum_210, getitem_1075, mul_2360, sum_212, getitem_1072, getitem_1069, mul_2351, sum_210, getitem_1066, getitem_1063, mul_2342, sum_208, getitem_1060, getitem_1057, mul_2333, sum_206, getitem_1051, mul_2324, sum_204, getitem_1048, getitem_1045, mul_2315, sum_200, mul_2306, sum_200, getitem_1039, getitem_1036, mul_2297, sum_198, getitem_1033, mul_2288, sum_196, getitem_1030, getitem_1027, mul_2279, sum_194, getitem_1024, getitem_1021, mul_2270, sum_192, getitem_1018, getitem_1015, mul_2261, sum_190, getitem_1012, getitem_1009, mul_2252, sum_188, getitem_1006, getitem_1003, mul_2243, sum_186, getitem_1000, getitem_997, mul_2234, sum_180, getitem_994, getitem_991, mul_2225, sum_182, getitem_988, getitem_985, mul_2216, sum_180, getitem_982, getitem_979, mul_2207, sum_178, getitem_976, getitem_973, mul_2198, sum_176, getitem_970, getitem_967, mul_2189, sum_174, getitem_964, getitem_961, mul_2180, sum_172, getitem_958, mul_2171, sum_170, getitem_955, mul_2162, sum_168, getitem_952, getitem_949, mul_2153, sum_166, getitem_946, getitem_943, mul_2144, sum_164, getitem_940, getitem_937, mul_2135, sum_162, getitem_934, getitem_931, mul_2126, sum_160, getitem_928, getitem_925, mul_2117, sum_158, getitem_922, getitem_919, mul_2108, sum_152, getitem_916, getitem_913, mul_2099, sum_154, getitem_910, getitem_907, mul_2090, sum_152, getitem_904, getitem_901, mul_2081, sum_150, getitem_898, getitem_895, mul_2072, sum_148, getitem_892, getitem_889, mul_2063, sum_146, getitem_886, getitem_883, mul_2054, sum_144, getitem_880, mul_2045, sum_142, getitem_877, mul_2036, sum_140, getitem_874, getitem_871, mul_2027, sum_138, getitem_868, getitem_865, mul_2018, sum_136, getitem_862, getitem_859, mul_2009, sum_134, getitem_856, getitem_853, mul_2000, sum_132, getitem_850, getitem_847, mul_1991, sum_130, getitem_844, getitem_841, mul_1982, sum_124, getitem_838, getitem_835, mul_1973, sum_126, getitem_832, getitem_829, mul_1964, sum_124, getitem_826, getitem_823, mul_1955, sum_122, getitem_820, getitem_817, mul_1946, sum_120, getitem_814, getitem_811, mul_1937, sum_118, getitem_808, getitem_805, mul_1928, sum_116, getitem_802, mul_1919, sum_114, getitem_799, mul_1910, sum_112, getitem_793, mul_1901, sum_110, getitem_790, getitem_787, mul_1892, sum_108, getitem_781, mul_1883, sum_106, getitem_778, getitem_775, mul_1874, sum_104, getitem_769, mul_1865, sum_102, getitem_766, getitem_763, mul_1856, sum_96, getitem_757, mul_1847, sum_98, getitem_754, getitem_751, mul_1838, sum_96, getitem_748, getitem_745, mul_1829, sum_94, getitem_742, getitem_739, mul_1820, sum_92, getitem_733, mul_1811, sum_90, getitem_730, getitem_727, mul_1802, sum_86, mul_1793, sum_86, getitem_721, getitem_718, mul_1784, sum_84, getitem_715, mul_1775, sum_82, getitem_712, getitem_709, mul_1766, sum_80, getitem_706, getitem_703, mul_1757, sum_78, getitem_700, getitem_697, mul_1748, sum_76, getitem_694, getitem_691, mul_1739, sum_74, getitem_688, getitem_685, mul_1730, sum_72, getitem_682, getitem_679, mul_1721, sum_66, getitem_676, getitem_673, mul_1712, sum_68, getitem_670, getitem_667, mul_1703, sum_66, getitem_664, getitem_661, mul_1694, sum_64, getitem_658, getitem_655, mul_1685, sum_62, getitem_652, getitem_649, mul_1676, sum_60, getitem_646, getitem_643, mul_1667, sum_58, getitem_640, mul_1658, sum_56, getitem_637, mul_1649, sum_54, getitem_634, getitem_631, mul_1640, sum_52, getitem_628, getitem_625, mul_1631, sum_50, getitem_622, getitem_619, mul_1622, sum_48, getitem_616, getitem_613, mul_1613, sum_46, getitem_610, getitem_607, mul_1604, sum_44, getitem_604, getitem_601, mul_1595, sum_38, getitem_598, getitem_595, mul_1586, sum_40, getitem_592, getitem_589, mul_1577, sum_38, getitem_586, getitem_583, mul_1568, sum_36, getitem_580, getitem_577, mul_1559, sum_34, getitem_574, getitem_571, mul_1550, sum_32, getitem_568, getitem_565, mul_1541, sum_30, getitem_562, mul_1532, sum_28, getitem_559, mul_1523, sum_26, getitem_556, getitem_553, mul_1514, sum_24, getitem_550, getitem_547, mul_1505, sum_22, getitem_544, getitem_541, mul_1496, sum_20, getitem_538, getitem_535, mul_1487, sum_18, getitem_532, getitem_529, mul_1478, sum_16, getitem_526, getitem_523, mul_1469, sum_10, getitem_520, getitem_517, mul_1460, sum_12, getitem_514, getitem_511, mul_1451, sum_10, getitem_508, getitem_505, mul_1442, sum_8, getitem_502, getitem_499, mul_1433, sum_6, getitem_496, getitem_493, mul_1424, sum_4, getitem_490, getitem_487, mul_1415, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    