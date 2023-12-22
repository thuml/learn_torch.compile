from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[512, 64, 1, 1]", primals_5: "f32[512]", primals_6: "f32[512]", primals_7: "f32[512, 16, 3, 3]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[256, 512, 1, 1]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[256, 64, 1, 1]", primals_14: "f32[256]", primals_15: "f32[256]", primals_16: "f32[512, 256, 1, 1]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512, 16, 3, 3]", primals_20: "f32[512]", primals_21: "f32[512]", primals_22: "f32[256, 512, 1, 1]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[512, 256, 1, 1]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512, 16, 3, 3]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[256, 512, 1, 1]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[1024, 256, 1, 1]", primals_35: "f32[1024]", primals_36: "f32[1024]", primals_37: "f32[1024, 32, 3, 3]", primals_38: "f32[1024]", primals_39: "f32[1024]", primals_40: "f32[512, 1024, 1, 1]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512, 256, 1, 1]", primals_44: "f32[512]", primals_45: "f32[512]", primals_46: "f32[1024, 512, 1, 1]", primals_47: "f32[1024]", primals_48: "f32[1024]", primals_49: "f32[1024, 32, 3, 3]", primals_50: "f32[1024]", primals_51: "f32[1024]", primals_52: "f32[512, 1024, 1, 1]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[1024, 512, 1, 1]", primals_56: "f32[1024]", primals_57: "f32[1024]", primals_58: "f32[1024, 32, 3, 3]", primals_59: "f32[1024]", primals_60: "f32[1024]", primals_61: "f32[512, 1024, 1, 1]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[1024, 512, 1, 1]", primals_65: "f32[1024]", primals_66: "f32[1024]", primals_67: "f32[1024, 32, 3, 3]", primals_68: "f32[1024]", primals_69: "f32[1024]", primals_70: "f32[512, 1024, 1, 1]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[2048, 512, 1, 1]", primals_74: "f32[2048]", primals_75: "f32[2048]", primals_76: "f32[2048, 64, 3, 3]", primals_77: "f32[2048]", primals_78: "f32[2048]", primals_79: "f32[1024, 2048, 1, 1]", primals_80: "f32[1024]", primals_81: "f32[1024]", primals_82: "f32[1024, 512, 1, 1]", primals_83: "f32[1024]", primals_84: "f32[1024]", primals_85: "f32[2048, 1024, 1, 1]", primals_86: "f32[2048]", primals_87: "f32[2048]", primals_88: "f32[2048, 64, 3, 3]", primals_89: "f32[2048]", primals_90: "f32[2048]", primals_91: "f32[1024, 2048, 1, 1]", primals_92: "f32[1024]", primals_93: "f32[1024]", primals_94: "f32[2048, 1024, 1, 1]", primals_95: "f32[2048]", primals_96: "f32[2048]", primals_97: "f32[2048, 64, 3, 3]", primals_98: "f32[2048]", primals_99: "f32[2048]", primals_100: "f32[1024, 2048, 1, 1]", primals_101: "f32[1024]", primals_102: "f32[1024]", primals_103: "f32[2048, 1024, 1, 1]", primals_104: "f32[2048]", primals_105: "f32[2048]", primals_106: "f32[2048, 64, 3, 3]", primals_107: "f32[2048]", primals_108: "f32[2048]", primals_109: "f32[1024, 2048, 1, 1]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_112: "f32[2048, 1024, 1, 1]", primals_113: "f32[2048]", primals_114: "f32[2048]", primals_115: "f32[2048, 64, 3, 3]", primals_116: "f32[2048]", primals_117: "f32[2048]", primals_118: "f32[1024, 2048, 1, 1]", primals_119: "f32[1024]", primals_120: "f32[1024]", primals_121: "f32[2048, 1024, 1, 1]", primals_122: "f32[2048]", primals_123: "f32[2048]", primals_124: "f32[2048, 64, 3, 3]", primals_125: "f32[2048]", primals_126: "f32[2048]", primals_127: "f32[1024, 2048, 1, 1]", primals_128: "f32[1024]", primals_129: "f32[1024]", primals_130: "f32[2048, 1024, 1, 1]", primals_131: "f32[2048]", primals_132: "f32[2048]", primals_133: "f32[2048, 64, 3, 3]", primals_134: "f32[2048]", primals_135: "f32[2048]", primals_136: "f32[1024, 2048, 1, 1]", primals_137: "f32[1024]", primals_138: "f32[1024]", primals_139: "f32[2048, 1024, 1, 1]", primals_140: "f32[2048]", primals_141: "f32[2048]", primals_142: "f32[2048, 64, 3, 3]", primals_143: "f32[2048]", primals_144: "f32[2048]", primals_145: "f32[1024, 2048, 1, 1]", primals_146: "f32[1024]", primals_147: "f32[1024]", primals_148: "f32[2048, 1024, 1, 1]", primals_149: "f32[2048]", primals_150: "f32[2048]", primals_151: "f32[2048, 64, 3, 3]", primals_152: "f32[2048]", primals_153: "f32[2048]", primals_154: "f32[1024, 2048, 1, 1]", primals_155: "f32[1024]", primals_156: "f32[1024]", primals_157: "f32[2048, 1024, 1, 1]", primals_158: "f32[2048]", primals_159: "f32[2048]", primals_160: "f32[2048, 64, 3, 3]", primals_161: "f32[2048]", primals_162: "f32[2048]", primals_163: "f32[1024, 2048, 1, 1]", primals_164: "f32[1024]", primals_165: "f32[1024]", primals_166: "f32[2048, 1024, 1, 1]", primals_167: "f32[2048]", primals_168: "f32[2048]", primals_169: "f32[2048, 64, 3, 3]", primals_170: "f32[2048]", primals_171: "f32[2048]", primals_172: "f32[1024, 2048, 1, 1]", primals_173: "f32[1024]", primals_174: "f32[1024]", primals_175: "f32[2048, 1024, 1, 1]", primals_176: "f32[2048]", primals_177: "f32[2048]", primals_178: "f32[2048, 64, 3, 3]", primals_179: "f32[2048]", primals_180: "f32[2048]", primals_181: "f32[1024, 2048, 1, 1]", primals_182: "f32[1024]", primals_183: "f32[1024]", primals_184: "f32[2048, 1024, 1, 1]", primals_185: "f32[2048]", primals_186: "f32[2048]", primals_187: "f32[2048, 64, 3, 3]", primals_188: "f32[2048]", primals_189: "f32[2048]", primals_190: "f32[1024, 2048, 1, 1]", primals_191: "f32[1024]", primals_192: "f32[1024]", primals_193: "f32[2048, 1024, 1, 1]", primals_194: "f32[2048]", primals_195: "f32[2048]", primals_196: "f32[2048, 64, 3, 3]", primals_197: "f32[2048]", primals_198: "f32[2048]", primals_199: "f32[1024, 2048, 1, 1]", primals_200: "f32[1024]", primals_201: "f32[1024]", primals_202: "f32[2048, 1024, 1, 1]", primals_203: "f32[2048]", primals_204: "f32[2048]", primals_205: "f32[2048, 64, 3, 3]", primals_206: "f32[2048]", primals_207: "f32[2048]", primals_208: "f32[1024, 2048, 1, 1]", primals_209: "f32[1024]", primals_210: "f32[1024]", primals_211: "f32[2048, 1024, 1, 1]", primals_212: "f32[2048]", primals_213: "f32[2048]", primals_214: "f32[2048, 64, 3, 3]", primals_215: "f32[2048]", primals_216: "f32[2048]", primals_217: "f32[1024, 2048, 1, 1]", primals_218: "f32[1024]", primals_219: "f32[1024]", primals_220: "f32[2048, 1024, 1, 1]", primals_221: "f32[2048]", primals_222: "f32[2048]", primals_223: "f32[2048, 64, 3, 3]", primals_224: "f32[2048]", primals_225: "f32[2048]", primals_226: "f32[1024, 2048, 1, 1]", primals_227: "f32[1024]", primals_228: "f32[1024]", primals_229: "f32[2048, 1024, 1, 1]", primals_230: "f32[2048]", primals_231: "f32[2048]", primals_232: "f32[2048, 64, 3, 3]", primals_233: "f32[2048]", primals_234: "f32[2048]", primals_235: "f32[1024, 2048, 1, 1]", primals_236: "f32[1024]", primals_237: "f32[1024]", primals_238: "f32[2048, 1024, 1, 1]", primals_239: "f32[2048]", primals_240: "f32[2048]", primals_241: "f32[2048, 64, 3, 3]", primals_242: "f32[2048]", primals_243: "f32[2048]", primals_244: "f32[1024, 2048, 1, 1]", primals_245: "f32[1024]", primals_246: "f32[1024]", primals_247: "f32[2048, 1024, 1, 1]", primals_248: "f32[2048]", primals_249: "f32[2048]", primals_250: "f32[2048, 64, 3, 3]", primals_251: "f32[2048]", primals_252: "f32[2048]", primals_253: "f32[1024, 2048, 1, 1]", primals_254: "f32[1024]", primals_255: "f32[1024]", primals_256: "f32[2048, 1024, 1, 1]", primals_257: "f32[2048]", primals_258: "f32[2048]", primals_259: "f32[2048, 64, 3, 3]", primals_260: "f32[2048]", primals_261: "f32[2048]", primals_262: "f32[1024, 2048, 1, 1]", primals_263: "f32[1024]", primals_264: "f32[1024]", primals_265: "f32[2048, 1024, 1, 1]", primals_266: "f32[2048]", primals_267: "f32[2048]", primals_268: "f32[2048, 64, 3, 3]", primals_269: "f32[2048]", primals_270: "f32[2048]", primals_271: "f32[1024, 2048, 1, 1]", primals_272: "f32[1024]", primals_273: "f32[1024]", primals_274: "f32[2048, 1024, 1, 1]", primals_275: "f32[2048]", primals_276: "f32[2048]", primals_277: "f32[2048, 64, 3, 3]", primals_278: "f32[2048]", primals_279: "f32[2048]", primals_280: "f32[1024, 2048, 1, 1]", primals_281: "f32[1024]", primals_282: "f32[1024]", primals_283: "f32[4096, 1024, 1, 1]", primals_284: "f32[4096]", primals_285: "f32[4096]", primals_286: "f32[4096, 128, 3, 3]", primals_287: "f32[4096]", primals_288: "f32[4096]", primals_289: "f32[2048, 4096, 1, 1]", primals_290: "f32[2048]", primals_291: "f32[2048]", primals_292: "f32[2048, 1024, 1, 1]", primals_293: "f32[2048]", primals_294: "f32[2048]", primals_295: "f32[4096, 2048, 1, 1]", primals_296: "f32[4096]", primals_297: "f32[4096]", primals_298: "f32[4096, 128, 3, 3]", primals_299: "f32[4096]", primals_300: "f32[4096]", primals_301: "f32[2048, 4096, 1, 1]", primals_302: "f32[2048]", primals_303: "f32[2048]", primals_304: "f32[4096, 2048, 1, 1]", primals_305: "f32[4096]", primals_306: "f32[4096]", primals_307: "f32[4096, 128, 3, 3]", primals_308: "f32[4096]", primals_309: "f32[4096]", primals_310: "f32[2048, 4096, 1, 1]", primals_311: "f32[2048]", primals_312: "f32[2048]", primals_313: "f32[1000, 2048]", primals_314: "f32[1000]", primals_315: "f32[64]", primals_316: "f32[64]", primals_317: "i64[]", primals_318: "f32[512]", primals_319: "f32[512]", primals_320: "i64[]", primals_321: "f32[512]", primals_322: "f32[512]", primals_323: "i64[]", primals_324: "f32[256]", primals_325: "f32[256]", primals_326: "i64[]", primals_327: "f32[256]", primals_328: "f32[256]", primals_329: "i64[]", primals_330: "f32[512]", primals_331: "f32[512]", primals_332: "i64[]", primals_333: "f32[512]", primals_334: "f32[512]", primals_335: "i64[]", primals_336: "f32[256]", primals_337: "f32[256]", primals_338: "i64[]", primals_339: "f32[512]", primals_340: "f32[512]", primals_341: "i64[]", primals_342: "f32[512]", primals_343: "f32[512]", primals_344: "i64[]", primals_345: "f32[256]", primals_346: "f32[256]", primals_347: "i64[]", primals_348: "f32[1024]", primals_349: "f32[1024]", primals_350: "i64[]", primals_351: "f32[1024]", primals_352: "f32[1024]", primals_353: "i64[]", primals_354: "f32[512]", primals_355: "f32[512]", primals_356: "i64[]", primals_357: "f32[512]", primals_358: "f32[512]", primals_359: "i64[]", primals_360: "f32[1024]", primals_361: "f32[1024]", primals_362: "i64[]", primals_363: "f32[1024]", primals_364: "f32[1024]", primals_365: "i64[]", primals_366: "f32[512]", primals_367: "f32[512]", primals_368: "i64[]", primals_369: "f32[1024]", primals_370: "f32[1024]", primals_371: "i64[]", primals_372: "f32[1024]", primals_373: "f32[1024]", primals_374: "i64[]", primals_375: "f32[512]", primals_376: "f32[512]", primals_377: "i64[]", primals_378: "f32[1024]", primals_379: "f32[1024]", primals_380: "i64[]", primals_381: "f32[1024]", primals_382: "f32[1024]", primals_383: "i64[]", primals_384: "f32[512]", primals_385: "f32[512]", primals_386: "i64[]", primals_387: "f32[2048]", primals_388: "f32[2048]", primals_389: "i64[]", primals_390: "f32[2048]", primals_391: "f32[2048]", primals_392: "i64[]", primals_393: "f32[1024]", primals_394: "f32[1024]", primals_395: "i64[]", primals_396: "f32[1024]", primals_397: "f32[1024]", primals_398: "i64[]", primals_399: "f32[2048]", primals_400: "f32[2048]", primals_401: "i64[]", primals_402: "f32[2048]", primals_403: "f32[2048]", primals_404: "i64[]", primals_405: "f32[1024]", primals_406: "f32[1024]", primals_407: "i64[]", primals_408: "f32[2048]", primals_409: "f32[2048]", primals_410: "i64[]", primals_411: "f32[2048]", primals_412: "f32[2048]", primals_413: "i64[]", primals_414: "f32[1024]", primals_415: "f32[1024]", primals_416: "i64[]", primals_417: "f32[2048]", primals_418: "f32[2048]", primals_419: "i64[]", primals_420: "f32[2048]", primals_421: "f32[2048]", primals_422: "i64[]", primals_423: "f32[1024]", primals_424: "f32[1024]", primals_425: "i64[]", primals_426: "f32[2048]", primals_427: "f32[2048]", primals_428: "i64[]", primals_429: "f32[2048]", primals_430: "f32[2048]", primals_431: "i64[]", primals_432: "f32[1024]", primals_433: "f32[1024]", primals_434: "i64[]", primals_435: "f32[2048]", primals_436: "f32[2048]", primals_437: "i64[]", primals_438: "f32[2048]", primals_439: "f32[2048]", primals_440: "i64[]", primals_441: "f32[1024]", primals_442: "f32[1024]", primals_443: "i64[]", primals_444: "f32[2048]", primals_445: "f32[2048]", primals_446: "i64[]", primals_447: "f32[2048]", primals_448: "f32[2048]", primals_449: "i64[]", primals_450: "f32[1024]", primals_451: "f32[1024]", primals_452: "i64[]", primals_453: "f32[2048]", primals_454: "f32[2048]", primals_455: "i64[]", primals_456: "f32[2048]", primals_457: "f32[2048]", primals_458: "i64[]", primals_459: "f32[1024]", primals_460: "f32[1024]", primals_461: "i64[]", primals_462: "f32[2048]", primals_463: "f32[2048]", primals_464: "i64[]", primals_465: "f32[2048]", primals_466: "f32[2048]", primals_467: "i64[]", primals_468: "f32[1024]", primals_469: "f32[1024]", primals_470: "i64[]", primals_471: "f32[2048]", primals_472: "f32[2048]", primals_473: "i64[]", primals_474: "f32[2048]", primals_475: "f32[2048]", primals_476: "i64[]", primals_477: "f32[1024]", primals_478: "f32[1024]", primals_479: "i64[]", primals_480: "f32[2048]", primals_481: "f32[2048]", primals_482: "i64[]", primals_483: "f32[2048]", primals_484: "f32[2048]", primals_485: "i64[]", primals_486: "f32[1024]", primals_487: "f32[1024]", primals_488: "i64[]", primals_489: "f32[2048]", primals_490: "f32[2048]", primals_491: "i64[]", primals_492: "f32[2048]", primals_493: "f32[2048]", primals_494: "i64[]", primals_495: "f32[1024]", primals_496: "f32[1024]", primals_497: "i64[]", primals_498: "f32[2048]", primals_499: "f32[2048]", primals_500: "i64[]", primals_501: "f32[2048]", primals_502: "f32[2048]", primals_503: "i64[]", primals_504: "f32[1024]", primals_505: "f32[1024]", primals_506: "i64[]", primals_507: "f32[2048]", primals_508: "f32[2048]", primals_509: "i64[]", primals_510: "f32[2048]", primals_511: "f32[2048]", primals_512: "i64[]", primals_513: "f32[1024]", primals_514: "f32[1024]", primals_515: "i64[]", primals_516: "f32[2048]", primals_517: "f32[2048]", primals_518: "i64[]", primals_519: "f32[2048]", primals_520: "f32[2048]", primals_521: "i64[]", primals_522: "f32[1024]", primals_523: "f32[1024]", primals_524: "i64[]", primals_525: "f32[2048]", primals_526: "f32[2048]", primals_527: "i64[]", primals_528: "f32[2048]", primals_529: "f32[2048]", primals_530: "i64[]", primals_531: "f32[1024]", primals_532: "f32[1024]", primals_533: "i64[]", primals_534: "f32[2048]", primals_535: "f32[2048]", primals_536: "i64[]", primals_537: "f32[2048]", primals_538: "f32[2048]", primals_539: "i64[]", primals_540: "f32[1024]", primals_541: "f32[1024]", primals_542: "i64[]", primals_543: "f32[2048]", primals_544: "f32[2048]", primals_545: "i64[]", primals_546: "f32[2048]", primals_547: "f32[2048]", primals_548: "i64[]", primals_549: "f32[1024]", primals_550: "f32[1024]", primals_551: "i64[]", primals_552: "f32[2048]", primals_553: "f32[2048]", primals_554: "i64[]", primals_555: "f32[2048]", primals_556: "f32[2048]", primals_557: "i64[]", primals_558: "f32[1024]", primals_559: "f32[1024]", primals_560: "i64[]", primals_561: "f32[2048]", primals_562: "f32[2048]", primals_563: "i64[]", primals_564: "f32[2048]", primals_565: "f32[2048]", primals_566: "i64[]", primals_567: "f32[1024]", primals_568: "f32[1024]", primals_569: "i64[]", primals_570: "f32[2048]", primals_571: "f32[2048]", primals_572: "i64[]", primals_573: "f32[2048]", primals_574: "f32[2048]", primals_575: "i64[]", primals_576: "f32[1024]", primals_577: "f32[1024]", primals_578: "i64[]", primals_579: "f32[2048]", primals_580: "f32[2048]", primals_581: "i64[]", primals_582: "f32[2048]", primals_583: "f32[2048]", primals_584: "i64[]", primals_585: "f32[1024]", primals_586: "f32[1024]", primals_587: "i64[]", primals_588: "f32[2048]", primals_589: "f32[2048]", primals_590: "i64[]", primals_591: "f32[2048]", primals_592: "f32[2048]", primals_593: "i64[]", primals_594: "f32[1024]", primals_595: "f32[1024]", primals_596: "i64[]", primals_597: "f32[4096]", primals_598: "f32[4096]", primals_599: "i64[]", primals_600: "f32[4096]", primals_601: "f32[4096]", primals_602: "i64[]", primals_603: "f32[2048]", primals_604: "f32[2048]", primals_605: "i64[]", primals_606: "f32[2048]", primals_607: "f32[2048]", primals_608: "i64[]", primals_609: "f32[4096]", primals_610: "f32[4096]", primals_611: "i64[]", primals_612: "f32[4096]", primals_613: "f32[4096]", primals_614: "i64[]", primals_615: "f32[2048]", primals_616: "f32[2048]", primals_617: "i64[]", primals_618: "f32[4096]", primals_619: "f32[4096]", primals_620: "i64[]", primals_621: "f32[4096]", primals_622: "f32[4096]", primals_623: "i64[]", primals_624: "f32[2048]", primals_625: "f32[2048]", primals_626: "i64[]", primals_627: "f32[8, 3, 224, 224]"):
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_101: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_102: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_416: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_417: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_428: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_429: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_440: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_441: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_452: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_453: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_464: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_465: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_476: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_477: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_488: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_489: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_500: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_501: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_512: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_513: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_524: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_525: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_536: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_537: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_548: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_549: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_560: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_561: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_572: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_573: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_584: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_585: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_596: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_597: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_608: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_609: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_620: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_621: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_632: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_633: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_644: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_645: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_656: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_657: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_668: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_669: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_680: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_681: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_692: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_693: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_704: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_705: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_716: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_717: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_728: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_729: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_740: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_741: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_752: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_753: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_764: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_765: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_776: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_777: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_788: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_789: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_800: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_801: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_812: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_813: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_824: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_825: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_836: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_837: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_848: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_849: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_860: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_861: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_872: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_873: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_884: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_885: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_896: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_897: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_908: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_909: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_920: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_921: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_932: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_933: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_944: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_945: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_956: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_957: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_968: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_969: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_980: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_981: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_992: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_993: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1004: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_1005: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1016: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_1017: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1028: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_1029: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1040: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_1041: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1052: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_1053: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
    unsqueeze_1054: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1064: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_1065: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1076: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_1077: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1088: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_1089: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1100: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_1101: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1112: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_1113: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1124: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_1125: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1136: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_1137: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1148: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_1149: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
    unsqueeze_1150: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1160: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_1161: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1172: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_1173: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
    unsqueeze_1174: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1184: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_1185: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1196: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_1197: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
    unsqueeze_1198: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1208: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_1209: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1220: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_1221: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
    unsqueeze_1222: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1232: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_1233: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1244: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_1245: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1256: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_1257: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1268: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_1269: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1280: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1281: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 2);  unsqueeze_1280 = None
    unsqueeze_1282: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 3);  unsqueeze_1281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1292: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1293: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 2);  unsqueeze_1292 = None
    unsqueeze_1294: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 3);  unsqueeze_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1304: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1305: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 2);  unsqueeze_1304 = None
    unsqueeze_1306: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 3);  unsqueeze_1305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1316: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1317: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 2);  unsqueeze_1316 = None
    unsqueeze_1318: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 3);  unsqueeze_1317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1328: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1329: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 2);  unsqueeze_1328 = None
    unsqueeze_1330: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 3);  unsqueeze_1329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1340: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1341: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
    unsqueeze_1342: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1352: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1353: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 2);  unsqueeze_1352 = None
    unsqueeze_1354: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 3);  unsqueeze_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1364: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1365: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 2);  unsqueeze_1364 = None
    unsqueeze_1366: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 3);  unsqueeze_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1376: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1377: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 2);  unsqueeze_1376 = None
    unsqueeze_1378: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 3);  unsqueeze_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 2);  unsqueeze_1388 = None
    unsqueeze_1390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 3);  unsqueeze_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1400: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1401: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
    unsqueeze_1402: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1412: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1413: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
    unsqueeze_1414: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1424: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1425: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
    unsqueeze_1426: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1436: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1437: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 2);  unsqueeze_1436 = None
    unsqueeze_1438: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 3);  unsqueeze_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1448: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1449: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 2);  unsqueeze_1448 = None
    unsqueeze_1450: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 3);  unsqueeze_1449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1460: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1461: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 2);  unsqueeze_1460 = None
    unsqueeze_1462: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 3);  unsqueeze_1461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1472: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1473: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 2);  unsqueeze_1472 = None
    unsqueeze_1474: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 3);  unsqueeze_1473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1484: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1485: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
    unsqueeze_1486: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1496: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1497: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 2);  unsqueeze_1496 = None
    unsqueeze_1498: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 3);  unsqueeze_1497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1508: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1509: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 2);  unsqueeze_1508 = None
    unsqueeze_1510: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 3);  unsqueeze_1509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1520: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1521: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 2);  unsqueeze_1520 = None
    unsqueeze_1522: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 3);  unsqueeze_1521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1532: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1533: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 2);  unsqueeze_1532 = None
    unsqueeze_1534: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 3);  unsqueeze_1533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1544: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1545: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
    unsqueeze_1546: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1556: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1557: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
    unsqueeze_1558: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1568: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1569: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
    unsqueeze_1570: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1580: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1581: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 2);  unsqueeze_1580 = None
    unsqueeze_1582: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 3);  unsqueeze_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1592: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1593: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 2);  unsqueeze_1592 = None
    unsqueeze_1594: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 3);  unsqueeze_1593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1604: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1605: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 2);  unsqueeze_1604 = None
    unsqueeze_1606: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 3);  unsqueeze_1605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1616: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1617: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 2);  unsqueeze_1616 = None
    unsqueeze_1618: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 3);  unsqueeze_1617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1628: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1629: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 2);  unsqueeze_1628 = None
    unsqueeze_1630: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 3);  unsqueeze_1629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
    unsqueeze_1642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze_1652: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1653: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 2);  unsqueeze_1652 = None
    unsqueeze_1654: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 3);  unsqueeze_1653 = None
    
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
    return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_627, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, relu_43, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, relu_57, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, relu_64, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, relu_71, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, relu_78, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, relu_92, convolution_96, squeeze_289, convolution_97, squeeze_292, relu_93, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, view, permute_1, le, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, unsqueeze_1054, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1282, unsqueeze_1294, unsqueeze_1306, unsqueeze_1318, unsqueeze_1330, unsqueeze_1342, unsqueeze_1354, unsqueeze_1366, unsqueeze_1378, unsqueeze_1390, unsqueeze_1402, unsqueeze_1414, unsqueeze_1426, unsqueeze_1438, unsqueeze_1450, unsqueeze_1462, unsqueeze_1474, unsqueeze_1486, unsqueeze_1498, unsqueeze_1510, unsqueeze_1522, unsqueeze_1534, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1582, unsqueeze_1594, unsqueeze_1606, unsqueeze_1618, unsqueeze_1630, unsqueeze_1642, unsqueeze_1654]
    