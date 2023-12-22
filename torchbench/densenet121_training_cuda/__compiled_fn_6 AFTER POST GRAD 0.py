from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[64]", primals_6: "f32[128, 64, 1, 1]", primals_7: "f32[128]", primals_9: "f32[32, 128, 3, 3]", primals_10: "f32[96]", primals_12: "f32[128, 96, 1, 1]", primals_13: "f32[128]", primals_15: "f32[32, 128, 3, 3]", primals_16: "f32[128]", primals_18: "f32[128, 128, 1, 1]", primals_19: "f32[128]", primals_21: "f32[32, 128, 3, 3]", primals_22: "f32[160]", primals_24: "f32[128, 160, 1, 1]", primals_25: "f32[128]", primals_27: "f32[32, 128, 3, 3]", primals_28: "f32[192]", primals_30: "f32[128, 192, 1, 1]", primals_31: "f32[128]", primals_33: "f32[32, 128, 3, 3]", primals_34: "f32[224]", primals_36: "f32[128, 224, 1, 1]", primals_37: "f32[128]", primals_39: "f32[32, 128, 3, 3]", primals_40: "f32[256]", primals_42: "f32[128, 256, 1, 1]", primals_43: "f32[128]", primals_45: "f32[128, 128, 1, 1]", primals_46: "f32[128]", primals_48: "f32[32, 128, 3, 3]", primals_49: "f32[160]", primals_51: "f32[128, 160, 1, 1]", primals_52: "f32[128]", primals_54: "f32[32, 128, 3, 3]", primals_55: "f32[192]", primals_57: "f32[128, 192, 1, 1]", primals_58: "f32[128]", primals_60: "f32[32, 128, 3, 3]", primals_61: "f32[224]", primals_63: "f32[128, 224, 1, 1]", primals_64: "f32[128]", primals_66: "f32[32, 128, 3, 3]", primals_67: "f32[256]", primals_69: "f32[128, 256, 1, 1]", primals_70: "f32[128]", primals_72: "f32[32, 128, 3, 3]", primals_73: "f32[288]", primals_75: "f32[128, 288, 1, 1]", primals_76: "f32[128]", primals_78: "f32[32, 128, 3, 3]", primals_79: "f32[320]", primals_81: "f32[128, 320, 1, 1]", primals_82: "f32[128]", primals_84: "f32[32, 128, 3, 3]", primals_85: "f32[352]", primals_87: "f32[128, 352, 1, 1]", primals_88: "f32[128]", primals_90: "f32[32, 128, 3, 3]", primals_91: "f32[384]", primals_93: "f32[128, 384, 1, 1]", primals_94: "f32[128]", primals_96: "f32[32, 128, 3, 3]", primals_97: "f32[416]", primals_99: "f32[128, 416, 1, 1]", primals_100: "f32[128]", primals_102: "f32[32, 128, 3, 3]", primals_103: "f32[448]", primals_105: "f32[128, 448, 1, 1]", primals_106: "f32[128]", primals_108: "f32[32, 128, 3, 3]", primals_109: "f32[480]", primals_111: "f32[128, 480, 1, 1]", primals_112: "f32[128]", primals_114: "f32[32, 128, 3, 3]", primals_115: "f32[512]", primals_117: "f32[256, 512, 1, 1]", primals_118: "f32[256]", primals_120: "f32[128, 256, 1, 1]", primals_121: "f32[128]", primals_123: "f32[32, 128, 3, 3]", primals_124: "f32[288]", primals_126: "f32[128, 288, 1, 1]", primals_127: "f32[128]", primals_129: "f32[32, 128, 3, 3]", primals_130: "f32[320]", primals_132: "f32[128, 320, 1, 1]", primals_133: "f32[128]", primals_135: "f32[32, 128, 3, 3]", primals_136: "f32[352]", primals_138: "f32[128, 352, 1, 1]", primals_139: "f32[128]", primals_141: "f32[32, 128, 3, 3]", primals_142: "f32[384]", primals_144: "f32[128, 384, 1, 1]", primals_145: "f32[128]", primals_147: "f32[32, 128, 3, 3]", primals_148: "f32[416]", primals_150: "f32[128, 416, 1, 1]", primals_151: "f32[128]", primals_153: "f32[32, 128, 3, 3]", primals_154: "f32[448]", primals_156: "f32[128, 448, 1, 1]", primals_157: "f32[128]", primals_159: "f32[32, 128, 3, 3]", primals_160: "f32[480]", primals_162: "f32[128, 480, 1, 1]", primals_163: "f32[128]", primals_165: "f32[32, 128, 3, 3]", primals_166: "f32[512]", primals_168: "f32[128, 512, 1, 1]", primals_169: "f32[128]", primals_171: "f32[32, 128, 3, 3]", primals_172: "f32[544]", primals_174: "f32[128, 544, 1, 1]", primals_175: "f32[128]", primals_177: "f32[32, 128, 3, 3]", primals_178: "f32[576]", primals_180: "f32[128, 576, 1, 1]", primals_181: "f32[128]", primals_183: "f32[32, 128, 3, 3]", primals_184: "f32[608]", primals_186: "f32[128, 608, 1, 1]", primals_187: "f32[128]", primals_189: "f32[32, 128, 3, 3]", primals_190: "f32[640]", primals_192: "f32[128, 640, 1, 1]", primals_193: "f32[128]", primals_195: "f32[32, 128, 3, 3]", primals_196: "f32[672]", primals_198: "f32[128, 672, 1, 1]", primals_199: "f32[128]", primals_201: "f32[32, 128, 3, 3]", primals_202: "f32[704]", primals_204: "f32[128, 704, 1, 1]", primals_205: "f32[128]", primals_207: "f32[32, 128, 3, 3]", primals_208: "f32[736]", primals_210: "f32[128, 736, 1, 1]", primals_211: "f32[128]", primals_213: "f32[32, 128, 3, 3]", primals_214: "f32[768]", primals_216: "f32[128, 768, 1, 1]", primals_217: "f32[128]", primals_219: "f32[32, 128, 3, 3]", primals_220: "f32[800]", primals_222: "f32[128, 800, 1, 1]", primals_223: "f32[128]", primals_225: "f32[32, 128, 3, 3]", primals_226: "f32[832]", primals_228: "f32[128, 832, 1, 1]", primals_229: "f32[128]", primals_231: "f32[32, 128, 3, 3]", primals_232: "f32[864]", primals_234: "f32[128, 864, 1, 1]", primals_235: "f32[128]", primals_237: "f32[32, 128, 3, 3]", primals_238: "f32[896]", primals_240: "f32[128, 896, 1, 1]", primals_241: "f32[128]", primals_243: "f32[32, 128, 3, 3]", primals_244: "f32[928]", primals_246: "f32[128, 928, 1, 1]", primals_247: "f32[128]", primals_249: "f32[32, 128, 3, 3]", primals_250: "f32[960]", primals_252: "f32[128, 960, 1, 1]", primals_253: "f32[128]", primals_255: "f32[32, 128, 3, 3]", primals_256: "f32[992]", primals_258: "f32[128, 992, 1, 1]", primals_259: "f32[128]", primals_261: "f32[32, 128, 3, 3]", primals_262: "f32[1024]", primals_264: "f32[512, 1024, 1, 1]", primals_265: "f32[512]", primals_267: "f32[128, 512, 1, 1]", primals_268: "f32[128]", primals_270: "f32[32, 128, 3, 3]", primals_271: "f32[544]", primals_273: "f32[128, 544, 1, 1]", primals_274: "f32[128]", primals_276: "f32[32, 128, 3, 3]", primals_277: "f32[576]", primals_279: "f32[128, 576, 1, 1]", primals_280: "f32[128]", primals_282: "f32[32, 128, 3, 3]", primals_283: "f32[608]", primals_285: "f32[128, 608, 1, 1]", primals_286: "f32[128]", primals_288: "f32[32, 128, 3, 3]", primals_289: "f32[640]", primals_291: "f32[128, 640, 1, 1]", primals_292: "f32[128]", primals_294: "f32[32, 128, 3, 3]", primals_295: "f32[672]", primals_297: "f32[128, 672, 1, 1]", primals_298: "f32[128]", primals_300: "f32[32, 128, 3, 3]", primals_301: "f32[704]", primals_303: "f32[128, 704, 1, 1]", primals_304: "f32[128]", primals_306: "f32[32, 128, 3, 3]", primals_307: "f32[736]", primals_309: "f32[128, 736, 1, 1]", primals_310: "f32[128]", primals_312: "f32[32, 128, 3, 3]", primals_313: "f32[768]", primals_315: "f32[128, 768, 1, 1]", primals_316: "f32[128]", primals_318: "f32[32, 128, 3, 3]", primals_319: "f32[800]", primals_321: "f32[128, 800, 1, 1]", primals_322: "f32[128]", primals_324: "f32[32, 128, 3, 3]", primals_325: "f32[832]", primals_327: "f32[128, 832, 1, 1]", primals_328: "f32[128]", primals_330: "f32[32, 128, 3, 3]", primals_331: "f32[864]", primals_333: "f32[128, 864, 1, 1]", primals_334: "f32[128]", primals_336: "f32[32, 128, 3, 3]", primals_337: "f32[896]", primals_339: "f32[128, 896, 1, 1]", primals_340: "f32[128]", primals_342: "f32[32, 128, 3, 3]", primals_343: "f32[928]", primals_345: "f32[128, 928, 1, 1]", primals_346: "f32[128]", primals_348: "f32[32, 128, 3, 3]", primals_349: "f32[960]", primals_351: "f32[128, 960, 1, 1]", primals_352: "f32[128]", primals_354: "f32[32, 128, 3, 3]", primals_355: "f32[992]", primals_357: "f32[128, 992, 1, 1]", primals_358: "f32[128]", primals_360: "f32[32, 128, 3, 3]", primals_361: "f32[1024]", primals_365: "f32[64]", primals_366: "f32[64]", primals_369: "f32[64]", primals_371: "f32[128]", primals_372: "f32[128]", primals_374: "f32[96]", primals_375: "f32[96]", primals_377: "f32[128]", primals_378: "f32[128]", primals_380: "f32[128]", primals_381: "f32[128]", primals_383: "f32[128]", primals_384: "f32[128]", primals_386: "f32[160]", primals_387: "f32[160]", primals_389: "f32[128]", primals_390: "f32[128]", primals_392: "f32[192]", primals_393: "f32[192]", primals_395: "f32[128]", primals_396: "f32[128]", primals_398: "f32[224]", primals_399: "f32[224]", primals_401: "f32[128]", primals_402: "f32[128]", primals_404: "f32[256]", primals_405: "f32[256]", primals_407: "f32[128]", primals_408: "f32[128]", primals_410: "f32[128]", primals_411: "f32[128]", primals_413: "f32[160]", primals_414: "f32[160]", primals_416: "f32[128]", primals_417: "f32[128]", primals_419: "f32[192]", primals_420: "f32[192]", primals_422: "f32[128]", primals_423: "f32[128]", primals_425: "f32[224]", primals_426: "f32[224]", primals_428: "f32[128]", primals_429: "f32[128]", primals_431: "f32[256]", primals_432: "f32[256]", primals_434: "f32[128]", primals_435: "f32[128]", primals_437: "f32[288]", primals_438: "f32[288]", primals_440: "f32[128]", primals_441: "f32[128]", primals_443: "f32[320]", primals_444: "f32[320]", primals_446: "f32[128]", primals_447: "f32[128]", primals_449: "f32[352]", primals_450: "f32[352]", primals_452: "f32[128]", primals_453: "f32[128]", primals_455: "f32[384]", primals_456: "f32[384]", primals_458: "f32[128]", primals_459: "f32[128]", primals_461: "f32[416]", primals_462: "f32[416]", primals_464: "f32[128]", primals_465: "f32[128]", primals_467: "f32[448]", primals_468: "f32[448]", primals_470: "f32[128]", primals_471: "f32[128]", primals_473: "f32[480]", primals_474: "f32[480]", primals_476: "f32[128]", primals_477: "f32[128]", primals_479: "f32[512]", primals_480: "f32[512]", primals_482: "f32[256]", primals_483: "f32[256]", primals_485: "f32[128]", primals_486: "f32[128]", primals_488: "f32[288]", primals_489: "f32[288]", primals_491: "f32[128]", primals_492: "f32[128]", primals_494: "f32[320]", primals_495: "f32[320]", primals_497: "f32[128]", primals_498: "f32[128]", primals_500: "f32[352]", primals_501: "f32[352]", primals_503: "f32[128]", primals_504: "f32[128]", primals_506: "f32[384]", primals_507: "f32[384]", primals_509: "f32[128]", primals_510: "f32[128]", primals_512: "f32[416]", primals_513: "f32[416]", primals_515: "f32[128]", primals_516: "f32[128]", primals_518: "f32[448]", primals_519: "f32[448]", primals_521: "f32[128]", primals_522: "f32[128]", primals_524: "f32[480]", primals_525: "f32[480]", primals_527: "f32[128]", primals_528: "f32[128]", primals_530: "f32[512]", primals_531: "f32[512]", primals_533: "f32[128]", primals_534: "f32[128]", primals_536: "f32[544]", primals_537: "f32[544]", primals_539: "f32[128]", primals_540: "f32[128]", primals_542: "f32[576]", primals_543: "f32[576]", primals_545: "f32[128]", primals_546: "f32[128]", primals_548: "f32[608]", primals_549: "f32[608]", primals_551: "f32[128]", primals_552: "f32[128]", primals_554: "f32[640]", primals_555: "f32[640]", primals_557: "f32[128]", primals_558: "f32[128]", primals_560: "f32[672]", primals_561: "f32[672]", primals_563: "f32[128]", primals_564: "f32[128]", primals_566: "f32[704]", primals_567: "f32[704]", primals_569: "f32[128]", primals_570: "f32[128]", primals_572: "f32[736]", primals_573: "f32[736]", primals_575: "f32[128]", primals_576: "f32[128]", primals_578: "f32[768]", primals_579: "f32[768]", primals_581: "f32[128]", primals_582: "f32[128]", primals_584: "f32[800]", primals_585: "f32[800]", primals_587: "f32[128]", primals_588: "f32[128]", primals_590: "f32[832]", primals_591: "f32[832]", primals_593: "f32[128]", primals_594: "f32[128]", primals_596: "f32[864]", primals_597: "f32[864]", primals_599: "f32[128]", primals_600: "f32[128]", primals_602: "f32[896]", primals_603: "f32[896]", primals_605: "f32[128]", primals_606: "f32[128]", primals_608: "f32[928]", primals_609: "f32[928]", primals_611: "f32[128]", primals_612: "f32[128]", primals_614: "f32[960]", primals_615: "f32[960]", primals_617: "f32[128]", primals_618: "f32[128]", primals_620: "f32[992]", primals_621: "f32[992]", primals_623: "f32[128]", primals_624: "f32[128]", primals_626: "f32[1024]", primals_627: "f32[1024]", primals_629: "f32[512]", primals_630: "f32[512]", primals_632: "f32[128]", primals_633: "f32[128]", primals_635: "f32[544]", primals_636: "f32[544]", primals_638: "f32[128]", primals_639: "f32[128]", primals_641: "f32[576]", primals_642: "f32[576]", primals_644: "f32[128]", primals_645: "f32[128]", primals_647: "f32[608]", primals_648: "f32[608]", primals_650: "f32[128]", primals_651: "f32[128]", primals_653: "f32[640]", primals_654: "f32[640]", primals_656: "f32[128]", primals_657: "f32[128]", primals_659: "f32[672]", primals_660: "f32[672]", primals_662: "f32[128]", primals_663: "f32[128]", primals_665: "f32[704]", primals_666: "f32[704]", primals_668: "f32[128]", primals_669: "f32[128]", primals_671: "f32[736]", primals_672: "f32[736]", primals_674: "f32[128]", primals_675: "f32[128]", primals_677: "f32[768]", primals_678: "f32[768]", primals_680: "f32[128]", primals_681: "f32[128]", primals_683: "f32[800]", primals_684: "f32[800]", primals_686: "f32[128]", primals_687: "f32[128]", primals_689: "f32[832]", primals_690: "f32[832]", primals_692: "f32[128]", primals_693: "f32[128]", primals_695: "f32[864]", primals_696: "f32[864]", primals_698: "f32[128]", primals_699: "f32[128]", primals_701: "f32[896]", primals_702: "f32[896]", primals_704: "f32[128]", primals_705: "f32[128]", primals_707: "f32[928]", primals_708: "f32[928]", primals_710: "f32[128]", primals_711: "f32[128]", primals_713: "f32[960]", primals_714: "f32[960]", primals_716: "f32[128]", primals_717: "f32[128]", primals_719: "f32[992]", primals_720: "f32[992]", primals_722: "f32[128]", primals_723: "f32[128]", primals_725: "f32[1024]", primals_726: "f32[1024]", primals_728: "f32[4, 3, 224, 224]", convolution: "f32[4, 64, 112, 112]", relu: "f32[4, 64, 112, 112]", getitem_1: "i64[4, 64, 56, 56]", relu_1: "f32[4, 64, 56, 56]", convolution_1: "f32[4, 128, 56, 56]", relu_2: "f32[4, 128, 56, 56]", cat: "f32[4, 96, 56, 56]", relu_3: "f32[4, 96, 56, 56]", convolution_3: "f32[4, 128, 56, 56]", relu_4: "f32[4, 128, 56, 56]", cat_1: "f32[4, 128, 56, 56]", relu_5: "f32[4, 128, 56, 56]", convolution_5: "f32[4, 128, 56, 56]", relu_6: "f32[4, 128, 56, 56]", cat_2: "f32[4, 160, 56, 56]", relu_7: "f32[4, 160, 56, 56]", convolution_7: "f32[4, 128, 56, 56]", relu_8: "f32[4, 128, 56, 56]", cat_3: "f32[4, 192, 56, 56]", relu_9: "f32[4, 192, 56, 56]", convolution_9: "f32[4, 128, 56, 56]", relu_10: "f32[4, 128, 56, 56]", cat_4: "f32[4, 224, 56, 56]", relu_11: "f32[4, 224, 56, 56]", convolution_11: "f32[4, 128, 56, 56]", relu_12: "f32[4, 128, 56, 56]", cat_5: "f32[4, 256, 56, 56]", relu_13: "f32[4, 256, 56, 56]", convolution_13: "f32[4, 128, 56, 56]", avg_pool2d: "f32[4, 128, 28, 28]", relu_14: "f32[4, 128, 28, 28]", convolution_14: "f32[4, 128, 28, 28]", relu_15: "f32[4, 128, 28, 28]", cat_6: "f32[4, 160, 28, 28]", relu_16: "f32[4, 160, 28, 28]", convolution_16: "f32[4, 128, 28, 28]", relu_17: "f32[4, 128, 28, 28]", cat_7: "f32[4, 192, 28, 28]", relu_18: "f32[4, 192, 28, 28]", convolution_18: "f32[4, 128, 28, 28]", relu_19: "f32[4, 128, 28, 28]", cat_8: "f32[4, 224, 28, 28]", relu_20: "f32[4, 224, 28, 28]", convolution_20: "f32[4, 128, 28, 28]", relu_21: "f32[4, 128, 28, 28]", cat_9: "f32[4, 256, 28, 28]", relu_22: "f32[4, 256, 28, 28]", convolution_22: "f32[4, 128, 28, 28]", relu_23: "f32[4, 128, 28, 28]", cat_10: "f32[4, 288, 28, 28]", relu_24: "f32[4, 288, 28, 28]", convolution_24: "f32[4, 128, 28, 28]", relu_25: "f32[4, 128, 28, 28]", cat_11: "f32[4, 320, 28, 28]", relu_26: "f32[4, 320, 28, 28]", convolution_26: "f32[4, 128, 28, 28]", relu_27: "f32[4, 128, 28, 28]", cat_12: "f32[4, 352, 28, 28]", relu_28: "f32[4, 352, 28, 28]", convolution_28: "f32[4, 128, 28, 28]", relu_29: "f32[4, 128, 28, 28]", cat_13: "f32[4, 384, 28, 28]", relu_30: "f32[4, 384, 28, 28]", convolution_30: "f32[4, 128, 28, 28]", relu_31: "f32[4, 128, 28, 28]", cat_14: "f32[4, 416, 28, 28]", relu_32: "f32[4, 416, 28, 28]", convolution_32: "f32[4, 128, 28, 28]", relu_33: "f32[4, 128, 28, 28]", cat_15: "f32[4, 448, 28, 28]", relu_34: "f32[4, 448, 28, 28]", convolution_34: "f32[4, 128, 28, 28]", relu_35: "f32[4, 128, 28, 28]", cat_16: "f32[4, 480, 28, 28]", relu_36: "f32[4, 480, 28, 28]", convolution_36: "f32[4, 128, 28, 28]", relu_37: "f32[4, 128, 28, 28]", cat_17: "f32[4, 512, 28, 28]", relu_38: "f32[4, 512, 28, 28]", convolution_38: "f32[4, 256, 28, 28]", avg_pool2d_1: "f32[4, 256, 14, 14]", relu_39: "f32[4, 256, 14, 14]", convolution_39: "f32[4, 128, 14, 14]", relu_40: "f32[4, 128, 14, 14]", cat_18: "f32[4, 288, 14, 14]", relu_41: "f32[4, 288, 14, 14]", convolution_41: "f32[4, 128, 14, 14]", relu_42: "f32[4, 128, 14, 14]", cat_19: "f32[4, 320, 14, 14]", relu_43: "f32[4, 320, 14, 14]", convolution_43: "f32[4, 128, 14, 14]", relu_44: "f32[4, 128, 14, 14]", cat_20: "f32[4, 352, 14, 14]", relu_45: "f32[4, 352, 14, 14]", convolution_45: "f32[4, 128, 14, 14]", relu_46: "f32[4, 128, 14, 14]", cat_21: "f32[4, 384, 14, 14]", relu_47: "f32[4, 384, 14, 14]", convolution_47: "f32[4, 128, 14, 14]", relu_48: "f32[4, 128, 14, 14]", cat_22: "f32[4, 416, 14, 14]", relu_49: "f32[4, 416, 14, 14]", convolution_49: "f32[4, 128, 14, 14]", relu_50: "f32[4, 128, 14, 14]", cat_23: "f32[4, 448, 14, 14]", relu_51: "f32[4, 448, 14, 14]", convolution_51: "f32[4, 128, 14, 14]", relu_52: "f32[4, 128, 14, 14]", cat_24: "f32[4, 480, 14, 14]", relu_53: "f32[4, 480, 14, 14]", convolution_53: "f32[4, 128, 14, 14]", relu_54: "f32[4, 128, 14, 14]", cat_25: "f32[4, 512, 14, 14]", relu_55: "f32[4, 512, 14, 14]", convolution_55: "f32[4, 128, 14, 14]", relu_56: "f32[4, 128, 14, 14]", cat_26: "f32[4, 544, 14, 14]", relu_57: "f32[4, 544, 14, 14]", convolution_57: "f32[4, 128, 14, 14]", relu_58: "f32[4, 128, 14, 14]", cat_27: "f32[4, 576, 14, 14]", relu_59: "f32[4, 576, 14, 14]", convolution_59: "f32[4, 128, 14, 14]", relu_60: "f32[4, 128, 14, 14]", cat_28: "f32[4, 608, 14, 14]", relu_61: "f32[4, 608, 14, 14]", convolution_61: "f32[4, 128, 14, 14]", relu_62: "f32[4, 128, 14, 14]", cat_29: "f32[4, 640, 14, 14]", relu_63: "f32[4, 640, 14, 14]", convolution_63: "f32[4, 128, 14, 14]", relu_64: "f32[4, 128, 14, 14]", cat_30: "f32[4, 672, 14, 14]", relu_65: "f32[4, 672, 14, 14]", convolution_65: "f32[4, 128, 14, 14]", relu_66: "f32[4, 128, 14, 14]", cat_31: "f32[4, 704, 14, 14]", relu_67: "f32[4, 704, 14, 14]", convolution_67: "f32[4, 128, 14, 14]", relu_68: "f32[4, 128, 14, 14]", cat_32: "f32[4, 736, 14, 14]", relu_69: "f32[4, 736, 14, 14]", convolution_69: "f32[4, 128, 14, 14]", relu_70: "f32[4, 128, 14, 14]", cat_33: "f32[4, 768, 14, 14]", relu_71: "f32[4, 768, 14, 14]", convolution_71: "f32[4, 128, 14, 14]", relu_72: "f32[4, 128, 14, 14]", cat_34: "f32[4, 800, 14, 14]", relu_73: "f32[4, 800, 14, 14]", convolution_73: "f32[4, 128, 14, 14]", relu_74: "f32[4, 128, 14, 14]", cat_35: "f32[4, 832, 14, 14]", relu_75: "f32[4, 832, 14, 14]", convolution_75: "f32[4, 128, 14, 14]", relu_76: "f32[4, 128, 14, 14]", cat_36: "f32[4, 864, 14, 14]", relu_77: "f32[4, 864, 14, 14]", convolution_77: "f32[4, 128, 14, 14]", relu_78: "f32[4, 128, 14, 14]", cat_37: "f32[4, 896, 14, 14]", relu_79: "f32[4, 896, 14, 14]", convolution_79: "f32[4, 128, 14, 14]", relu_80: "f32[4, 128, 14, 14]", cat_38: "f32[4, 928, 14, 14]", relu_81: "f32[4, 928, 14, 14]", convolution_81: "f32[4, 128, 14, 14]", relu_82: "f32[4, 128, 14, 14]", cat_39: "f32[4, 960, 14, 14]", relu_83: "f32[4, 960, 14, 14]", convolution_83: "f32[4, 128, 14, 14]", relu_84: "f32[4, 128, 14, 14]", cat_40: "f32[4, 992, 14, 14]", relu_85: "f32[4, 992, 14, 14]", convolution_85: "f32[4, 128, 14, 14]", relu_86: "f32[4, 128, 14, 14]", cat_41: "f32[4, 1024, 14, 14]", relu_87: "f32[4, 1024, 14, 14]", convolution_87: "f32[4, 512, 14, 14]", avg_pool2d_2: "f32[4, 512, 7, 7]", relu_88: "f32[4, 512, 7, 7]", convolution_88: "f32[4, 128, 7, 7]", relu_89: "f32[4, 128, 7, 7]", cat_42: "f32[4, 544, 7, 7]", relu_90: "f32[4, 544, 7, 7]", convolution_90: "f32[4, 128, 7, 7]", relu_91: "f32[4, 128, 7, 7]", cat_43: "f32[4, 576, 7, 7]", relu_92: "f32[4, 576, 7, 7]", convolution_92: "f32[4, 128, 7, 7]", relu_93: "f32[4, 128, 7, 7]", cat_44: "f32[4, 608, 7, 7]", relu_94: "f32[4, 608, 7, 7]", convolution_94: "f32[4, 128, 7, 7]", relu_95: "f32[4, 128, 7, 7]", cat_45: "f32[4, 640, 7, 7]", relu_96: "f32[4, 640, 7, 7]", convolution_96: "f32[4, 128, 7, 7]", relu_97: "f32[4, 128, 7, 7]", cat_46: "f32[4, 672, 7, 7]", relu_98: "f32[4, 672, 7, 7]", convolution_98: "f32[4, 128, 7, 7]", relu_99: "f32[4, 128, 7, 7]", cat_47: "f32[4, 704, 7, 7]", relu_100: "f32[4, 704, 7, 7]", convolution_100: "f32[4, 128, 7, 7]", relu_101: "f32[4, 128, 7, 7]", cat_48: "f32[4, 736, 7, 7]", relu_102: "f32[4, 736, 7, 7]", convolution_102: "f32[4, 128, 7, 7]", relu_103: "f32[4, 128, 7, 7]", cat_49: "f32[4, 768, 7, 7]", relu_104: "f32[4, 768, 7, 7]", convolution_104: "f32[4, 128, 7, 7]", relu_105: "f32[4, 128, 7, 7]", cat_50: "f32[4, 800, 7, 7]", relu_106: "f32[4, 800, 7, 7]", convolution_106: "f32[4, 128, 7, 7]", relu_107: "f32[4, 128, 7, 7]", cat_51: "f32[4, 832, 7, 7]", relu_108: "f32[4, 832, 7, 7]", convolution_108: "f32[4, 128, 7, 7]", relu_109: "f32[4, 128, 7, 7]", cat_52: "f32[4, 864, 7, 7]", relu_110: "f32[4, 864, 7, 7]", convolution_110: "f32[4, 128, 7, 7]", relu_111: "f32[4, 128, 7, 7]", cat_53: "f32[4, 896, 7, 7]", relu_112: "f32[4, 896, 7, 7]", convolution_112: "f32[4, 128, 7, 7]", relu_113: "f32[4, 128, 7, 7]", cat_54: "f32[4, 928, 7, 7]", relu_114: "f32[4, 928, 7, 7]", convolution_114: "f32[4, 128, 7, 7]", relu_115: "f32[4, 128, 7, 7]", cat_55: "f32[4, 960, 7, 7]", relu_116: "f32[4, 960, 7, 7]", convolution_116: "f32[4, 128, 7, 7]", relu_117: "f32[4, 128, 7, 7]", cat_56: "f32[4, 992, 7, 7]", relu_118: "f32[4, 992, 7, 7]", convolution_118: "f32[4, 128, 7, 7]", relu_119: "f32[4, 128, 7, 7]", cat_57: "f32[4, 1024, 7, 7]", view: "f32[4, 1024]", permute_1: "f32[1000, 1024]", le: "b8[4, 1024, 7, 7]", sub_240: "f32[4, 64, 56, 56]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:217, code: out = self.classifier(out)
    mm: "f32[4, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:216, code: out = torch.flatten(out, 1)
    view_2: "f32[4, 1024, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:215, code: out = F.adaptive_avg_pool2d(out, (1, 1))
    expand: "f32[4, 1024, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 1024, 7, 7]);  view_2 = None
    div: "f32[4, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:214, code: out = F.relu(features, inplace=True)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    add_242: "f32[1024]" = torch.ops.aten.add.Tensor(primals_726, 1e-05);  primals_726 = None
    rsqrt: "f32[1024]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    unsqueeze_968: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_725, 0);  primals_725 = None
    unsqueeze_969: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_121: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_970);  cat_57 = unsqueeze_970 = None
    mul_363: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_121);  sub_121 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
    mul_368: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt, primals_361);  primals_361 = None
    unsqueeze_977: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_978: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_369: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_979);  where = unsqueeze_979 = None
    mul_370: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    slice_1: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 0, 512)
    slice_2: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 512, 544)
    slice_3: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 544, 576)
    slice_4: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 576, 608)
    slice_5: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 608, 640)
    slice_6: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 640, 672)
    slice_7: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 672, 704)
    slice_8: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 704, 736)
    slice_9: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 736, 768)
    slice_10: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 768, 800)
    slice_11: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 800, 832)
    slice_12: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 832, 864)
    slice_13: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 864, 896)
    slice_14: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 896, 928)
    slice_15: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 928, 960)
    slice_16: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 960, 992)
    slice_17: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_369, 1, 992, 1024);  mul_369 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward = torch.ops.aten.convolution_backward.default(slice_17, relu_119, primals_360, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_17 = primals_360 = None
    getitem_2: "f32[4, 128, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[32, 128, 3, 3]" = convolution_backward[1];  convolution_backward = None
    le_1: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_119, 0);  relu_119 = None
    where_1: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_2);  le_1 = getitem_2 = None
    add_243: "f32[128]" = torch.ops.aten.add.Tensor(primals_723, 1e-05);  primals_723 = None
    rsqrt_1: "f32[128]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
    unsqueeze_980: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_722, 0);  primals_722 = None
    unsqueeze_981: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_122: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_982);  convolution_118 = unsqueeze_982 = None
    mul_371: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_122);  sub_122 = None
    sum_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
    mul_376: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_358);  primals_358 = None
    unsqueeze_989: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
    unsqueeze_990: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_377: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_991);  where_1 = unsqueeze_991 = None
    mul_378: "f32[128]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_377, relu_118, primals_357, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_377 = primals_357 = None
    getitem_5: "f32[4, 992, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[128, 992, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    le_2: "b8[4, 992, 7, 7]" = torch.ops.aten.le.Scalar(relu_118, 0);  relu_118 = None
    where_2: "f32[4, 992, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_5);  le_2 = getitem_5 = None
    add_244: "f32[992]" = torch.ops.aten.add.Tensor(primals_720, 1e-05);  primals_720 = None
    rsqrt_2: "f32[992]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    unsqueeze_992: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(primals_719, 0);  primals_719 = None
    unsqueeze_993: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    sum_6: "f32[992]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_123: "f32[4, 992, 7, 7]" = torch.ops.aten.sub.Tensor(cat_56, unsqueeze_994);  cat_56 = unsqueeze_994 = None
    mul_379: "f32[4, 992, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_123);  sub_123 = None
    sum_7: "f32[992]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 2, 3]);  mul_379 = None
    mul_384: "f32[992]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_355);  primals_355 = None
    unsqueeze_1001: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
    unsqueeze_1002: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_385: "f32[4, 992, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_1003);  where_2 = unsqueeze_1003 = None
    mul_386: "f32[992]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_18: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 0, 512)
    slice_19: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 512, 544)
    slice_20: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 544, 576)
    slice_21: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 576, 608)
    slice_22: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 608, 640)
    slice_23: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 640, 672)
    slice_24: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 672, 704)
    slice_25: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 704, 736)
    slice_26: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 736, 768)
    slice_27: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 768, 800)
    slice_28: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 800, 832)
    slice_29: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 832, 864)
    slice_30: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 864, 896)
    slice_31: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 896, 928)
    slice_32: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 928, 960)
    slice_33: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_385, 1, 960, 992);  mul_385 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_245: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, slice_18);  slice_1 = slice_18 = None
    add_246: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, slice_19);  slice_2 = slice_19 = None
    add_247: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, slice_20);  slice_3 = slice_20 = None
    add_248: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_4, slice_21);  slice_4 = slice_21 = None
    add_249: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, slice_22);  slice_5 = slice_22 = None
    add_250: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_6, slice_23);  slice_6 = slice_23 = None
    add_251: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_7, slice_24);  slice_7 = slice_24 = None
    add_252: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_8, slice_25);  slice_8 = slice_25 = None
    add_253: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_9, slice_26);  slice_9 = slice_26 = None
    add_254: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_10, slice_27);  slice_10 = slice_27 = None
    add_255: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_11, slice_28);  slice_11 = slice_28 = None
    add_256: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_12, slice_29);  slice_12 = slice_29 = None
    add_257: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_13, slice_30);  slice_13 = slice_30 = None
    add_258: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_14, slice_31);  slice_14 = slice_31 = None
    add_259: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_15, slice_32);  slice_15 = slice_32 = None
    add_260: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_16, slice_33);  slice_16 = slice_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(add_260, relu_117, primals_354, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_260 = primals_354 = None
    getitem_8: "f32[4, 128, 7, 7]" = convolution_backward_2[0]
    getitem_9: "f32[32, 128, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    le_3: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_117, 0);  relu_117 = None
    where_3: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_8);  le_3 = getitem_8 = None
    add_261: "f32[128]" = torch.ops.aten.add.Tensor(primals_717, 1e-05);  primals_717 = None
    rsqrt_3: "f32[128]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    unsqueeze_1004: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_716, 0);  primals_716 = None
    unsqueeze_1005: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    sum_8: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_124: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_1006);  convolution_116 = unsqueeze_1006 = None
    mul_387: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_124);  sub_124 = None
    sum_9: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3]);  mul_387 = None
    mul_392: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_352);  primals_352 = None
    unsqueeze_1013: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_1014: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_393: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_1015);  where_3 = unsqueeze_1015 = None
    mul_394: "f32[128]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_393, relu_116, primals_351, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_393 = primals_351 = None
    getitem_11: "f32[4, 960, 7, 7]" = convolution_backward_3[0]
    getitem_12: "f32[128, 960, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    le_4: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(relu_116, 0);  relu_116 = None
    where_4: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_11);  le_4 = getitem_11 = None
    add_262: "f32[960]" = torch.ops.aten.add.Tensor(primals_714, 1e-05);  primals_714 = None
    rsqrt_4: "f32[960]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    unsqueeze_1016: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_713, 0);  primals_713 = None
    unsqueeze_1017: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    sum_10: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_125: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_1018);  cat_55 = unsqueeze_1018 = None
    mul_395: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_125);  sub_125 = None
    sum_11: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3]);  mul_395 = None
    mul_400: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_349);  primals_349 = None
    unsqueeze_1025: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_1026: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_401: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_1027);  where_4 = unsqueeze_1027 = None
    mul_402: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_34: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 0, 512)
    slice_35: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 512, 544)
    slice_36: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 544, 576)
    slice_37: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 576, 608)
    slice_38: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 608, 640)
    slice_39: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 640, 672)
    slice_40: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 672, 704)
    slice_41: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 704, 736)
    slice_42: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 736, 768)
    slice_43: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 768, 800)
    slice_44: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 800, 832)
    slice_45: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 832, 864)
    slice_46: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 864, 896)
    slice_47: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 896, 928)
    slice_48: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_401, 1, 928, 960);  mul_401 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_263: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_245, slice_34);  add_245 = slice_34 = None
    add_264: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_246, slice_35);  add_246 = slice_35 = None
    add_265: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_247, slice_36);  add_247 = slice_36 = None
    add_266: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_248, slice_37);  add_248 = slice_37 = None
    add_267: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_249, slice_38);  add_249 = slice_38 = None
    add_268: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_250, slice_39);  add_250 = slice_39 = None
    add_269: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_251, slice_40);  add_251 = slice_40 = None
    add_270: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_252, slice_41);  add_252 = slice_41 = None
    add_271: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_253, slice_42);  add_253 = slice_42 = None
    add_272: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_254, slice_43);  add_254 = slice_43 = None
    add_273: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_255, slice_44);  add_255 = slice_44 = None
    add_274: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_256, slice_45);  add_256 = slice_45 = None
    add_275: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_257, slice_46);  add_257 = slice_46 = None
    add_276: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_258, slice_47);  add_258 = slice_47 = None
    add_277: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_259, slice_48);  add_259 = slice_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_277, relu_115, primals_348, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_277 = primals_348 = None
    getitem_14: "f32[4, 128, 7, 7]" = convolution_backward_4[0]
    getitem_15: "f32[32, 128, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    le_5: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_115, 0);  relu_115 = None
    where_5: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_14);  le_5 = getitem_14 = None
    add_278: "f32[128]" = torch.ops.aten.add.Tensor(primals_711, 1e-05);  primals_711 = None
    rsqrt_5: "f32[128]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    unsqueeze_1028: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_710, 0);  primals_710 = None
    unsqueeze_1029: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    sum_12: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_126: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_1030);  convolution_114 = unsqueeze_1030 = None
    mul_403: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_126);  sub_126 = None
    sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 2, 3]);  mul_403 = None
    mul_408: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_346);  primals_346 = None
    unsqueeze_1037: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_1038: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_409: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_1039);  where_5 = unsqueeze_1039 = None
    mul_410: "f32[128]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_409, relu_114, primals_345, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_409 = primals_345 = None
    getitem_17: "f32[4, 928, 7, 7]" = convolution_backward_5[0]
    getitem_18: "f32[128, 928, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    le_6: "b8[4, 928, 7, 7]" = torch.ops.aten.le.Scalar(relu_114, 0);  relu_114 = None
    where_6: "f32[4, 928, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_17);  le_6 = getitem_17 = None
    add_279: "f32[928]" = torch.ops.aten.add.Tensor(primals_708, 1e-05);  primals_708 = None
    rsqrt_6: "f32[928]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
    unsqueeze_1040: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(primals_707, 0);  primals_707 = None
    unsqueeze_1041: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    sum_14: "f32[928]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_127: "f32[4, 928, 7, 7]" = torch.ops.aten.sub.Tensor(cat_54, unsqueeze_1042);  cat_54 = unsqueeze_1042 = None
    mul_411: "f32[4, 928, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_127);  sub_127 = None
    sum_15: "f32[928]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3]);  mul_411 = None
    mul_416: "f32[928]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_343);  primals_343 = None
    unsqueeze_1049: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
    unsqueeze_1050: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    mul_417: "f32[4, 928, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_1051);  where_6 = unsqueeze_1051 = None
    mul_418: "f32[928]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_49: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 0, 512)
    slice_50: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 512, 544)
    slice_51: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 544, 576)
    slice_52: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 576, 608)
    slice_53: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 608, 640)
    slice_54: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 640, 672)
    slice_55: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 672, 704)
    slice_56: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 704, 736)
    slice_57: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 736, 768)
    slice_58: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 768, 800)
    slice_59: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 800, 832)
    slice_60: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 832, 864)
    slice_61: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 864, 896)
    slice_62: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_417, 1, 896, 928);  mul_417 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_280: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_263, slice_49);  add_263 = slice_49 = None
    add_281: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_264, slice_50);  add_264 = slice_50 = None
    add_282: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_265, slice_51);  add_265 = slice_51 = None
    add_283: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_266, slice_52);  add_266 = slice_52 = None
    add_284: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_267, slice_53);  add_267 = slice_53 = None
    add_285: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_268, slice_54);  add_268 = slice_54 = None
    add_286: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_269, slice_55);  add_269 = slice_55 = None
    add_287: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_270, slice_56);  add_270 = slice_56 = None
    add_288: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_271, slice_57);  add_271 = slice_57 = None
    add_289: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_272, slice_58);  add_272 = slice_58 = None
    add_290: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_273, slice_59);  add_273 = slice_59 = None
    add_291: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_274, slice_60);  add_274 = slice_60 = None
    add_292: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_275, slice_61);  add_275 = slice_61 = None
    add_293: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_276, slice_62);  add_276 = slice_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_293, relu_113, primals_342, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_293 = primals_342 = None
    getitem_20: "f32[4, 128, 7, 7]" = convolution_backward_6[0]
    getitem_21: "f32[32, 128, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    le_7: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_113, 0);  relu_113 = None
    where_7: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_20);  le_7 = getitem_20 = None
    add_294: "f32[128]" = torch.ops.aten.add.Tensor(primals_705, 1e-05);  primals_705 = None
    rsqrt_7: "f32[128]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    unsqueeze_1052: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_704, 0);  primals_704 = None
    unsqueeze_1053: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
    unsqueeze_1054: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_128: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_1054);  convolution_112 = unsqueeze_1054 = None
    mul_419: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_128);  sub_128 = None
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
    mul_424: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_340);  primals_340 = None
    unsqueeze_1061: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
    unsqueeze_1062: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    mul_425: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_1063);  where_7 = unsqueeze_1063 = None
    mul_426: "f32[128]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_425, relu_112, primals_339, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = primals_339 = None
    getitem_23: "f32[4, 896, 7, 7]" = convolution_backward_7[0]
    getitem_24: "f32[128, 896, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    le_8: "b8[4, 896, 7, 7]" = torch.ops.aten.le.Scalar(relu_112, 0);  relu_112 = None
    where_8: "f32[4, 896, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, getitem_23);  le_8 = getitem_23 = None
    add_295: "f32[896]" = torch.ops.aten.add.Tensor(primals_702, 1e-05);  primals_702 = None
    rsqrt_8: "f32[896]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    unsqueeze_1064: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_701, 0);  primals_701 = None
    unsqueeze_1065: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
    sum_18: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_129: "f32[4, 896, 7, 7]" = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_1066);  cat_53 = unsqueeze_1066 = None
    mul_427: "f32[4, 896, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_129);  sub_129 = None
    sum_19: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_432: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_337);  primals_337 = None
    unsqueeze_1073: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_1074: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    mul_433: "f32[4, 896, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_1075);  where_8 = unsqueeze_1075 = None
    mul_434: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_63: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 0, 512)
    slice_64: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 512, 544)
    slice_65: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 544, 576)
    slice_66: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 576, 608)
    slice_67: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 608, 640)
    slice_68: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 640, 672)
    slice_69: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 672, 704)
    slice_70: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 704, 736)
    slice_71: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 736, 768)
    slice_72: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 768, 800)
    slice_73: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 800, 832)
    slice_74: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 832, 864)
    slice_75: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_433, 1, 864, 896);  mul_433 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_296: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_280, slice_63);  add_280 = slice_63 = None
    add_297: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_281, slice_64);  add_281 = slice_64 = None
    add_298: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_282, slice_65);  add_282 = slice_65 = None
    add_299: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_283, slice_66);  add_283 = slice_66 = None
    add_300: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_284, slice_67);  add_284 = slice_67 = None
    add_301: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_285, slice_68);  add_285 = slice_68 = None
    add_302: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_286, slice_69);  add_286 = slice_69 = None
    add_303: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_287, slice_70);  add_287 = slice_70 = None
    add_304: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_288, slice_71);  add_288 = slice_71 = None
    add_305: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_289, slice_72);  add_289 = slice_72 = None
    add_306: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_290, slice_73);  add_290 = slice_73 = None
    add_307: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_291, slice_74);  add_291 = slice_74 = None
    add_308: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_292, slice_75);  add_292 = slice_75 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(add_308, relu_111, primals_336, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_308 = primals_336 = None
    getitem_26: "f32[4, 128, 7, 7]" = convolution_backward_8[0]
    getitem_27: "f32[32, 128, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    le_9: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_111, 0);  relu_111 = None
    where_9: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, getitem_26);  le_9 = getitem_26 = None
    add_309: "f32[128]" = torch.ops.aten.add.Tensor(primals_699, 1e-05);  primals_699 = None
    rsqrt_9: "f32[128]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
    unsqueeze_1076: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_698, 0);  primals_698 = None
    unsqueeze_1077: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_130: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_1078);  convolution_110 = unsqueeze_1078 = None
    mul_435: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_130);  sub_130 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
    mul_440: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_334);  primals_334 = None
    unsqueeze_1085: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_1086: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    mul_441: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_1087);  where_9 = unsqueeze_1087 = None
    mul_442: "f32[128]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_441, relu_110, primals_333, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_441 = primals_333 = None
    getitem_29: "f32[4, 864, 7, 7]" = convolution_backward_9[0]
    getitem_30: "f32[128, 864, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    le_10: "b8[4, 864, 7, 7]" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
    where_10: "f32[4, 864, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, getitem_29);  le_10 = getitem_29 = None
    add_310: "f32[864]" = torch.ops.aten.add.Tensor(primals_696, 1e-05);  primals_696 = None
    rsqrt_10: "f32[864]" = torch.ops.aten.rsqrt.default(add_310);  add_310 = None
    unsqueeze_1088: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(primals_695, 0);  primals_695 = None
    unsqueeze_1089: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None
    sum_22: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_131: "f32[4, 864, 7, 7]" = torch.ops.aten.sub.Tensor(cat_52, unsqueeze_1090);  cat_52 = unsqueeze_1090 = None
    mul_443: "f32[4, 864, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_131);  sub_131 = None
    sum_23: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 2, 3]);  mul_443 = None
    mul_448: "f32[864]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_331);  primals_331 = None
    unsqueeze_1097: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_1098: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_449: "f32[4, 864, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_1099);  where_10 = unsqueeze_1099 = None
    mul_450: "f32[864]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_76: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 0, 512)
    slice_77: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 512, 544)
    slice_78: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 544, 576)
    slice_79: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 576, 608)
    slice_80: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 608, 640)
    slice_81: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 640, 672)
    slice_82: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 672, 704)
    slice_83: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 704, 736)
    slice_84: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 736, 768)
    slice_85: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 768, 800)
    slice_86: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 800, 832)
    slice_87: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_449, 1, 832, 864);  mul_449 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_311: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_296, slice_76);  add_296 = slice_76 = None
    add_312: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_297, slice_77);  add_297 = slice_77 = None
    add_313: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_298, slice_78);  add_298 = slice_78 = None
    add_314: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_299, slice_79);  add_299 = slice_79 = None
    add_315: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_300, slice_80);  add_300 = slice_80 = None
    add_316: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_301, slice_81);  add_301 = slice_81 = None
    add_317: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_302, slice_82);  add_302 = slice_82 = None
    add_318: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_303, slice_83);  add_303 = slice_83 = None
    add_319: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_304, slice_84);  add_304 = slice_84 = None
    add_320: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_305, slice_85);  add_305 = slice_85 = None
    add_321: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_306, slice_86);  add_306 = slice_86 = None
    add_322: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_307, slice_87);  add_307 = slice_87 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_322, relu_109, primals_330, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_322 = primals_330 = None
    getitem_32: "f32[4, 128, 7, 7]" = convolution_backward_10[0]
    getitem_33: "f32[32, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    le_11: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
    where_11: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, getitem_32);  le_11 = getitem_32 = None
    add_323: "f32[128]" = torch.ops.aten.add.Tensor(primals_693, 1e-05);  primals_693 = None
    rsqrt_11: "f32[128]" = torch.ops.aten.rsqrt.default(add_323);  add_323 = None
    unsqueeze_1100: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_692, 0);  primals_692 = None
    unsqueeze_1101: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_132: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_1102);  convolution_108 = unsqueeze_1102 = None
    mul_451: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_132);  sub_132 = None
    sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
    mul_456: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_328);  primals_328 = None
    unsqueeze_1109: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_1110: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_457: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_1111);  where_11 = unsqueeze_1111 = None
    mul_458: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_457, relu_108, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_457 = primals_327 = None
    getitem_35: "f32[4, 832, 7, 7]" = convolution_backward_11[0]
    getitem_36: "f32[128, 832, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    le_12: "b8[4, 832, 7, 7]" = torch.ops.aten.le.Scalar(relu_108, 0);  relu_108 = None
    where_12: "f32[4, 832, 7, 7]" = torch.ops.aten.where.self(le_12, full_default, getitem_35);  le_12 = getitem_35 = None
    add_324: "f32[832]" = torch.ops.aten.add.Tensor(primals_690, 1e-05);  primals_690 = None
    rsqrt_12: "f32[832]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
    unsqueeze_1112: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(primals_689, 0);  primals_689 = None
    unsqueeze_1113: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    sum_26: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_133: "f32[4, 832, 7, 7]" = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_1114);  cat_51 = unsqueeze_1114 = None
    mul_459: "f32[4, 832, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_133);  sub_133 = None
    sum_27: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 2, 3]);  mul_459 = None
    mul_464: "f32[832]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_325);  primals_325 = None
    unsqueeze_1121: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_1122: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_465: "f32[4, 832, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_1123);  where_12 = unsqueeze_1123 = None
    mul_466: "f32[832]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_88: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 0, 512)
    slice_89: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 512, 544)
    slice_90: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 544, 576)
    slice_91: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 576, 608)
    slice_92: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 608, 640)
    slice_93: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 640, 672)
    slice_94: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 672, 704)
    slice_95: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 704, 736)
    slice_96: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 736, 768)
    slice_97: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 768, 800)
    slice_98: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_465, 1, 800, 832);  mul_465 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_325: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_311, slice_88);  add_311 = slice_88 = None
    add_326: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_312, slice_89);  add_312 = slice_89 = None
    add_327: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_313, slice_90);  add_313 = slice_90 = None
    add_328: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_314, slice_91);  add_314 = slice_91 = None
    add_329: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_315, slice_92);  add_315 = slice_92 = None
    add_330: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_316, slice_93);  add_316 = slice_93 = None
    add_331: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_317, slice_94);  add_317 = slice_94 = None
    add_332: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_318, slice_95);  add_318 = slice_95 = None
    add_333: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_319, slice_96);  add_319 = slice_96 = None
    add_334: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_320, slice_97);  add_320 = slice_97 = None
    add_335: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_321, slice_98);  add_321 = slice_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_335, relu_107, primals_324, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_335 = primals_324 = None
    getitem_38: "f32[4, 128, 7, 7]" = convolution_backward_12[0]
    getitem_39: "f32[32, 128, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    le_13: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
    where_13: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_13, full_default, getitem_38);  le_13 = getitem_38 = None
    add_336: "f32[128]" = torch.ops.aten.add.Tensor(primals_687, 1e-05);  primals_687 = None
    rsqrt_13: "f32[128]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
    unsqueeze_1124: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_686, 0);  primals_686 = None
    unsqueeze_1125: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    sum_28: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_134: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_1126);  convolution_106 = unsqueeze_1126 = None
    mul_467: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_134);  sub_134 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_472: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_322);  primals_322 = None
    unsqueeze_1133: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_1134: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    mul_473: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_1135);  where_13 = unsqueeze_1135 = None
    mul_474: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_473, relu_106, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_473 = primals_321 = None
    getitem_41: "f32[4, 800, 7, 7]" = convolution_backward_13[0]
    getitem_42: "f32[128, 800, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    le_14: "b8[4, 800, 7, 7]" = torch.ops.aten.le.Scalar(relu_106, 0);  relu_106 = None
    where_14: "f32[4, 800, 7, 7]" = torch.ops.aten.where.self(le_14, full_default, getitem_41);  le_14 = getitem_41 = None
    add_337: "f32[800]" = torch.ops.aten.add.Tensor(primals_684, 1e-05);  primals_684 = None
    rsqrt_14: "f32[800]" = torch.ops.aten.rsqrt.default(add_337);  add_337 = None
    unsqueeze_1136: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(primals_683, 0);  primals_683 = None
    unsqueeze_1137: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None
    sum_30: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_135: "f32[4, 800, 7, 7]" = torch.ops.aten.sub.Tensor(cat_50, unsqueeze_1138);  cat_50 = unsqueeze_1138 = None
    mul_475: "f32[4, 800, 7, 7]" = torch.ops.aten.mul.Tensor(where_14, sub_135);  sub_135 = None
    sum_31: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 2, 3]);  mul_475 = None
    mul_480: "f32[800]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_319);  primals_319 = None
    unsqueeze_1145: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_1146: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    mul_481: "f32[4, 800, 7, 7]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_1147);  where_14 = unsqueeze_1147 = None
    mul_482: "f32[800]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_99: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 0, 512)
    slice_100: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 512, 544)
    slice_101: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 544, 576)
    slice_102: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 576, 608)
    slice_103: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 608, 640)
    slice_104: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 640, 672)
    slice_105: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 672, 704)
    slice_106: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 704, 736)
    slice_107: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 736, 768)
    slice_108: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_481, 1, 768, 800);  mul_481 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_338: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_325, slice_99);  add_325 = slice_99 = None
    add_339: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_326, slice_100);  add_326 = slice_100 = None
    add_340: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_327, slice_101);  add_327 = slice_101 = None
    add_341: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_328, slice_102);  add_328 = slice_102 = None
    add_342: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_329, slice_103);  add_329 = slice_103 = None
    add_343: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_330, slice_104);  add_330 = slice_104 = None
    add_344: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_331, slice_105);  add_331 = slice_105 = None
    add_345: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_332, slice_106);  add_332 = slice_106 = None
    add_346: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_333, slice_107);  add_333 = slice_107 = None
    add_347: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_334, slice_108);  add_334 = slice_108 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_347, relu_105, primals_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_347 = primals_318 = None
    getitem_44: "f32[4, 128, 7, 7]" = convolution_backward_14[0]
    getitem_45: "f32[32, 128, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    le_15: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
    where_15: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_15, full_default, getitem_44);  le_15 = getitem_44 = None
    add_348: "f32[128]" = torch.ops.aten.add.Tensor(primals_681, 1e-05);  primals_681 = None
    rsqrt_15: "f32[128]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
    unsqueeze_1148: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_680, 0);  primals_680 = None
    unsqueeze_1149: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
    unsqueeze_1150: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None
    sum_32: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_136: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_1150);  convolution_104 = unsqueeze_1150 = None
    mul_483: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_136);  sub_136 = None
    sum_33: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2, 3]);  mul_483 = None
    mul_488: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_316);  primals_316 = None
    unsqueeze_1157: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_1158: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    mul_489: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_1159);  where_15 = unsqueeze_1159 = None
    mul_490: "f32[128]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_489, relu_104, primals_315, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_489 = primals_315 = None
    getitem_47: "f32[4, 768, 7, 7]" = convolution_backward_15[0]
    getitem_48: "f32[128, 768, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    le_16: "b8[4, 768, 7, 7]" = torch.ops.aten.le.Scalar(relu_104, 0);  relu_104 = None
    where_16: "f32[4, 768, 7, 7]" = torch.ops.aten.where.self(le_16, full_default, getitem_47);  le_16 = getitem_47 = None
    add_349: "f32[768]" = torch.ops.aten.add.Tensor(primals_678, 1e-05);  primals_678 = None
    rsqrt_16: "f32[768]" = torch.ops.aten.rsqrt.default(add_349);  add_349 = None
    unsqueeze_1160: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_677, 0);  primals_677 = None
    unsqueeze_1161: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_137: "f32[4, 768, 7, 7]" = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_1162);  cat_49 = unsqueeze_1162 = None
    mul_491: "f32[4, 768, 7, 7]" = torch.ops.aten.mul.Tensor(where_16, sub_137);  sub_137 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_496: "f32[768]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_313);  primals_313 = None
    unsqueeze_1169: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_1170: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    mul_497: "f32[4, 768, 7, 7]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_1171);  where_16 = unsqueeze_1171 = None
    mul_498: "f32[768]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_109: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 0, 512)
    slice_110: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 512, 544)
    slice_111: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 544, 576)
    slice_112: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 576, 608)
    slice_113: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 608, 640)
    slice_114: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 640, 672)
    slice_115: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 672, 704)
    slice_116: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 704, 736)
    slice_117: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_497, 1, 736, 768);  mul_497 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_350: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_338, slice_109);  add_338 = slice_109 = None
    add_351: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_339, slice_110);  add_339 = slice_110 = None
    add_352: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_340, slice_111);  add_340 = slice_111 = None
    add_353: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_341, slice_112);  add_341 = slice_112 = None
    add_354: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_342, slice_113);  add_342 = slice_113 = None
    add_355: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_343, slice_114);  add_343 = slice_114 = None
    add_356: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_344, slice_115);  add_344 = slice_115 = None
    add_357: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_345, slice_116);  add_345 = slice_116 = None
    add_358: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_346, slice_117);  add_346 = slice_117 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(add_358, relu_103, primals_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_358 = primals_312 = None
    getitem_50: "f32[4, 128, 7, 7]" = convolution_backward_16[0]
    getitem_51: "f32[32, 128, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    le_17: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
    where_17: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_17, full_default, getitem_50);  le_17 = getitem_50 = None
    add_359: "f32[128]" = torch.ops.aten.add.Tensor(primals_675, 1e-05);  primals_675 = None
    rsqrt_17: "f32[128]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
    unsqueeze_1172: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_674, 0);  primals_674 = None
    unsqueeze_1173: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
    unsqueeze_1174: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None
    sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_138: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_1174);  convolution_102 = unsqueeze_1174 = None
    mul_499: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_17, sub_138);  sub_138 = None
    sum_37: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_504: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_310);  primals_310 = None
    unsqueeze_1181: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_1182: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    mul_505: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_1183);  where_17 = unsqueeze_1183 = None
    mul_506: "f32[128]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_505, relu_102, primals_309, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = primals_309 = None
    getitem_53: "f32[4, 736, 7, 7]" = convolution_backward_17[0]
    getitem_54: "f32[128, 736, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    le_18: "b8[4, 736, 7, 7]" = torch.ops.aten.le.Scalar(relu_102, 0);  relu_102 = None
    where_18: "f32[4, 736, 7, 7]" = torch.ops.aten.where.self(le_18, full_default, getitem_53);  le_18 = getitem_53 = None
    add_360: "f32[736]" = torch.ops.aten.add.Tensor(primals_672, 1e-05);  primals_672 = None
    rsqrt_18: "f32[736]" = torch.ops.aten.rsqrt.default(add_360);  add_360 = None
    unsqueeze_1184: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(primals_671, 0);  primals_671 = None
    unsqueeze_1185: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    sum_38: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_139: "f32[4, 736, 7, 7]" = torch.ops.aten.sub.Tensor(cat_48, unsqueeze_1186);  cat_48 = unsqueeze_1186 = None
    mul_507: "f32[4, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, sub_139);  sub_139 = None
    sum_39: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_512: "f32[736]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_307);  primals_307 = None
    unsqueeze_1193: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_1194: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    mul_513: "f32[4, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_1195);  where_18 = unsqueeze_1195 = None
    mul_514: "f32[736]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_118: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 0, 512)
    slice_119: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 512, 544)
    slice_120: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 544, 576)
    slice_121: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 576, 608)
    slice_122: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 608, 640)
    slice_123: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 640, 672)
    slice_124: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 672, 704)
    slice_125: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_513, 1, 704, 736);  mul_513 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_361: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_350, slice_118);  add_350 = slice_118 = None
    add_362: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_351, slice_119);  add_351 = slice_119 = None
    add_363: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_352, slice_120);  add_352 = slice_120 = None
    add_364: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_353, slice_121);  add_353 = slice_121 = None
    add_365: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_354, slice_122);  add_354 = slice_122 = None
    add_366: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_355, slice_123);  add_355 = slice_123 = None
    add_367: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_356, slice_124);  add_356 = slice_124 = None
    add_368: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_357, slice_125);  add_357 = slice_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(add_368, relu_101, primals_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_368 = primals_306 = None
    getitem_56: "f32[4, 128, 7, 7]" = convolution_backward_18[0]
    getitem_57: "f32[32, 128, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    le_19: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_101, 0);  relu_101 = None
    where_19: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_19, full_default, getitem_56);  le_19 = getitem_56 = None
    add_369: "f32[128]" = torch.ops.aten.add.Tensor(primals_669, 1e-05);  primals_669 = None
    rsqrt_19: "f32[128]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
    unsqueeze_1196: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_668, 0);  primals_668 = None
    unsqueeze_1197: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
    unsqueeze_1198: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
    sum_40: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_140: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_1198);  convolution_100 = unsqueeze_1198 = None
    mul_515: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_140);  sub_140 = None
    sum_41: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
    mul_520: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_304);  primals_304 = None
    unsqueeze_1205: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_1206: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    mul_521: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_1207);  where_19 = unsqueeze_1207 = None
    mul_522: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_521, relu_100, primals_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_521 = primals_303 = None
    getitem_59: "f32[4, 704, 7, 7]" = convolution_backward_19[0]
    getitem_60: "f32[128, 704, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    le_20: "b8[4, 704, 7, 7]" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
    where_20: "f32[4, 704, 7, 7]" = torch.ops.aten.where.self(le_20, full_default, getitem_59);  le_20 = getitem_59 = None
    add_370: "f32[704]" = torch.ops.aten.add.Tensor(primals_666, 1e-05);  primals_666 = None
    rsqrt_20: "f32[704]" = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
    unsqueeze_1208: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(primals_665, 0);  primals_665 = None
    unsqueeze_1209: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None
    sum_42: "f32[704]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_141: "f32[4, 704, 7, 7]" = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_1210);  cat_47 = unsqueeze_1210 = None
    mul_523: "f32[4, 704, 7, 7]" = torch.ops.aten.mul.Tensor(where_20, sub_141);  sub_141 = None
    sum_43: "f32[704]" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 2, 3]);  mul_523 = None
    mul_528: "f32[704]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_301);  primals_301 = None
    unsqueeze_1217: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_1218: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    mul_529: "f32[4, 704, 7, 7]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_1219);  where_20 = unsqueeze_1219 = None
    mul_530: "f32[704]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_126: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 0, 512)
    slice_127: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 512, 544)
    slice_128: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 544, 576)
    slice_129: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 576, 608)
    slice_130: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 608, 640)
    slice_131: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 640, 672)
    slice_132: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_529, 1, 672, 704);  mul_529 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_371: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_361, slice_126);  add_361 = slice_126 = None
    add_372: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_362, slice_127);  add_362 = slice_127 = None
    add_373: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_363, slice_128);  add_363 = slice_128 = None
    add_374: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_364, slice_129);  add_364 = slice_129 = None
    add_375: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_365, slice_130);  add_365 = slice_130 = None
    add_376: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_366, slice_131);  add_366 = slice_131 = None
    add_377: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_367, slice_132);  add_367 = slice_132 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(add_377, relu_99, primals_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_377 = primals_300 = None
    getitem_62: "f32[4, 128, 7, 7]" = convolution_backward_20[0]
    getitem_63: "f32[32, 128, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    le_21: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_99, 0);  relu_99 = None
    where_21: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_21, full_default, getitem_62);  le_21 = getitem_62 = None
    add_378: "f32[128]" = torch.ops.aten.add.Tensor(primals_663, 1e-05);  primals_663 = None
    rsqrt_21: "f32[128]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
    unsqueeze_1220: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_662, 0);  primals_662 = None
    unsqueeze_1221: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
    unsqueeze_1222: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None
    sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_142: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_1222);  convolution_98 = unsqueeze_1222 = None
    mul_531: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_21, sub_142);  sub_142 = None
    sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None
    mul_536: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_298);  primals_298 = None
    unsqueeze_1229: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_1230: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    mul_537: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_1231);  where_21 = unsqueeze_1231 = None
    mul_538: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_537, relu_98, primals_297, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_537 = primals_297 = None
    getitem_65: "f32[4, 672, 7, 7]" = convolution_backward_21[0]
    getitem_66: "f32[128, 672, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    le_22: "b8[4, 672, 7, 7]" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
    where_22: "f32[4, 672, 7, 7]" = torch.ops.aten.where.self(le_22, full_default, getitem_65);  le_22 = getitem_65 = None
    add_379: "f32[672]" = torch.ops.aten.add.Tensor(primals_660, 1e-05);  primals_660 = None
    rsqrt_22: "f32[672]" = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
    unsqueeze_1232: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_659, 0);  primals_659 = None
    unsqueeze_1233: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None
    sum_46: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_143: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(cat_46, unsqueeze_1234);  cat_46 = unsqueeze_1234 = None
    mul_539: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_143);  sub_143 = None
    sum_47: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
    mul_544: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_295);  primals_295 = None
    unsqueeze_1241: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_1242: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    mul_545: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_1243);  where_22 = unsqueeze_1243 = None
    mul_546: "f32[672]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_133: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 0, 512)
    slice_134: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 512, 544)
    slice_135: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 544, 576)
    slice_136: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 576, 608)
    slice_137: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 608, 640)
    slice_138: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 640, 672);  mul_545 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_380: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_371, slice_133);  add_371 = slice_133 = None
    add_381: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_372, slice_134);  add_372 = slice_134 = None
    add_382: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_373, slice_135);  add_373 = slice_135 = None
    add_383: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_374, slice_136);  add_374 = slice_136 = None
    add_384: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_375, slice_137);  add_375 = slice_137 = None
    add_385: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_376, slice_138);  add_376 = slice_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_385, relu_97, primals_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_385 = primals_294 = None
    getitem_68: "f32[4, 128, 7, 7]" = convolution_backward_22[0]
    getitem_69: "f32[32, 128, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    le_23: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
    where_23: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_23, full_default, getitem_68);  le_23 = getitem_68 = None
    add_386: "f32[128]" = torch.ops.aten.add.Tensor(primals_657, 1e-05);  primals_657 = None
    rsqrt_23: "f32[128]" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
    unsqueeze_1244: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_656, 0);  primals_656 = None
    unsqueeze_1245: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_144: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_1246);  convolution_96 = unsqueeze_1246 = None
    mul_547: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_23, sub_144);  sub_144 = None
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_552: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_292);  primals_292 = None
    unsqueeze_1253: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_1254: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_553: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_1255);  where_23 = unsqueeze_1255 = None
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_553, relu_96, primals_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_553 = primals_291 = None
    getitem_71: "f32[4, 640, 7, 7]" = convolution_backward_23[0]
    getitem_72: "f32[128, 640, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    le_24: "b8[4, 640, 7, 7]" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
    where_24: "f32[4, 640, 7, 7]" = torch.ops.aten.where.self(le_24, full_default, getitem_71);  le_24 = getitem_71 = None
    add_387: "f32[640]" = torch.ops.aten.add.Tensor(primals_654, 1e-05);  primals_654 = None
    rsqrt_24: "f32[640]" = torch.ops.aten.rsqrt.default(add_387);  add_387 = None
    unsqueeze_1256: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_653, 0);  primals_653 = None
    unsqueeze_1257: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    sum_50: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_145: "f32[4, 640, 7, 7]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_1258);  cat_45 = unsqueeze_1258 = None
    mul_555: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_145);  sub_145 = None
    sum_51: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
    mul_560: "f32[640]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_289);  primals_289 = None
    unsqueeze_1265: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_1266: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_561: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_1267);  where_24 = unsqueeze_1267 = None
    mul_562: "f32[640]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_139: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_561, 1, 0, 512)
    slice_140: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_561, 1, 512, 544)
    slice_141: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_561, 1, 544, 576)
    slice_142: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_561, 1, 576, 608)
    slice_143: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_561, 1, 608, 640);  mul_561 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_388: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_380, slice_139);  add_380 = slice_139 = None
    add_389: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_381, slice_140);  add_381 = slice_140 = None
    add_390: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_382, slice_141);  add_382 = slice_141 = None
    add_391: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_383, slice_142);  add_383 = slice_142 = None
    add_392: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_384, slice_143);  add_384 = slice_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(add_392, relu_95, primals_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_392 = primals_288 = None
    getitem_74: "f32[4, 128, 7, 7]" = convolution_backward_24[0]
    getitem_75: "f32[32, 128, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    le_25: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
    where_25: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_25, full_default, getitem_74);  le_25 = getitem_74 = None
    add_393: "f32[128]" = torch.ops.aten.add.Tensor(primals_651, 1e-05);  primals_651 = None
    rsqrt_25: "f32[128]" = torch.ops.aten.rsqrt.default(add_393);  add_393 = None
    unsqueeze_1268: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_650, 0);  primals_650 = None
    unsqueeze_1269: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    sum_52: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_146: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_1270);  convolution_94 = unsqueeze_1270 = None
    mul_563: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_25, sub_146);  sub_146 = None
    sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_568: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_286);  primals_286 = None
    unsqueeze_1277: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_1278: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_569: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_1279);  where_25 = unsqueeze_1279 = None
    mul_570: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_569, relu_94, primals_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_569 = primals_285 = None
    getitem_77: "f32[4, 608, 7, 7]" = convolution_backward_25[0]
    getitem_78: "f32[128, 608, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    le_26: "b8[4, 608, 7, 7]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
    where_26: "f32[4, 608, 7, 7]" = torch.ops.aten.where.self(le_26, full_default, getitem_77);  le_26 = getitem_77 = None
    add_394: "f32[608]" = torch.ops.aten.add.Tensor(primals_648, 1e-05);  primals_648 = None
    rsqrt_26: "f32[608]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
    unsqueeze_1280: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(primals_647, 0);  primals_647 = None
    unsqueeze_1281: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 2);  unsqueeze_1280 = None
    unsqueeze_1282: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 3);  unsqueeze_1281 = None
    sum_54: "f32[608]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_147: "f32[4, 608, 7, 7]" = torch.ops.aten.sub.Tensor(cat_44, unsqueeze_1282);  cat_44 = unsqueeze_1282 = None
    mul_571: "f32[4, 608, 7, 7]" = torch.ops.aten.mul.Tensor(where_26, sub_147);  sub_147 = None
    sum_55: "f32[608]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_576: "f32[608]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_283);  primals_283 = None
    unsqueeze_1289: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_1290: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
    unsqueeze_1291: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
    mul_577: "f32[4, 608, 7, 7]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_1291);  where_26 = unsqueeze_1291 = None
    mul_578: "f32[608]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_144: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_577, 1, 0, 512)
    slice_145: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_577, 1, 512, 544)
    slice_146: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_577, 1, 544, 576)
    slice_147: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_577, 1, 576, 608);  mul_577 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_395: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_388, slice_144);  add_388 = slice_144 = None
    add_396: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_389, slice_145);  add_389 = slice_145 = None
    add_397: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_390, slice_146);  add_390 = slice_146 = None
    add_398: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_391, slice_147);  add_391 = slice_147 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(add_398, relu_93, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_398 = primals_282 = None
    getitem_80: "f32[4, 128, 7, 7]" = convolution_backward_26[0]
    getitem_81: "f32[32, 128, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    le_27: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
    where_27: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_27, full_default, getitem_80);  le_27 = getitem_80 = None
    add_399: "f32[128]" = torch.ops.aten.add.Tensor(primals_645, 1e-05);  primals_645 = None
    rsqrt_27: "f32[128]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
    unsqueeze_1292: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_644, 0);  primals_644 = None
    unsqueeze_1293: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 2);  unsqueeze_1292 = None
    unsqueeze_1294: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 3);  unsqueeze_1293 = None
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_148: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_1294);  convolution_92 = unsqueeze_1294 = None
    mul_579: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_27, sub_148);  sub_148 = None
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
    mul_584: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_280);  primals_280 = None
    unsqueeze_1301: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_1302: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
    unsqueeze_1303: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
    mul_585: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_1303);  where_27 = unsqueeze_1303 = None
    mul_586: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_585, relu_92, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_585 = primals_279 = None
    getitem_83: "f32[4, 576, 7, 7]" = convolution_backward_27[0]
    getitem_84: "f32[128, 576, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    le_28: "b8[4, 576, 7, 7]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
    where_28: "f32[4, 576, 7, 7]" = torch.ops.aten.where.self(le_28, full_default, getitem_83);  le_28 = getitem_83 = None
    add_400: "f32[576]" = torch.ops.aten.add.Tensor(primals_642, 1e-05);  primals_642 = None
    rsqrt_28: "f32[576]" = torch.ops.aten.rsqrt.default(add_400);  add_400 = None
    unsqueeze_1304: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_641, 0);  primals_641 = None
    unsqueeze_1305: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 2);  unsqueeze_1304 = None
    unsqueeze_1306: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 3);  unsqueeze_1305 = None
    sum_58: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_149: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_1306);  cat_43 = unsqueeze_1306 = None
    mul_587: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_28, sub_149);  sub_149 = None
    sum_59: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
    mul_592: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_277);  primals_277 = None
    unsqueeze_1313: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_1314: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
    unsqueeze_1315: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
    mul_593: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_1315);  where_28 = unsqueeze_1315 = None
    mul_594: "f32[576]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_148: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_593, 1, 0, 512)
    slice_149: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_593, 1, 512, 544)
    slice_150: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_593, 1, 544, 576);  mul_593 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_401: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_395, slice_148);  add_395 = slice_148 = None
    add_402: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_396, slice_149);  add_396 = slice_149 = None
    add_403: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_397, slice_150);  add_397 = slice_150 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_403, relu_91, primals_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_403 = primals_276 = None
    getitem_86: "f32[4, 128, 7, 7]" = convolution_backward_28[0]
    getitem_87: "f32[32, 128, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    le_29: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
    where_29: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_29, full_default, getitem_86);  le_29 = getitem_86 = None
    add_404: "f32[128]" = torch.ops.aten.add.Tensor(primals_639, 1e-05);  primals_639 = None
    rsqrt_29: "f32[128]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
    unsqueeze_1316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_638, 0);  primals_638 = None
    unsqueeze_1317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 2);  unsqueeze_1316 = None
    unsqueeze_1318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 3);  unsqueeze_1317 = None
    sum_60: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_150: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_1318);  convolution_90 = unsqueeze_1318 = None
    mul_595: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_29, sub_150);  sub_150 = None
    sum_61: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3]);  mul_595 = None
    mul_600: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_274);  primals_274 = None
    unsqueeze_1325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_1326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
    unsqueeze_1327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
    mul_601: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_1327);  where_29 = unsqueeze_1327 = None
    mul_602: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_601, relu_90, primals_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_601 = primals_273 = None
    getitem_89: "f32[4, 544, 7, 7]" = convolution_backward_29[0]
    getitem_90: "f32[128, 544, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    le_30: "b8[4, 544, 7, 7]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
    where_30: "f32[4, 544, 7, 7]" = torch.ops.aten.where.self(le_30, full_default, getitem_89);  le_30 = getitem_89 = None
    add_405: "f32[544]" = torch.ops.aten.add.Tensor(primals_636, 1e-05);  primals_636 = None
    rsqrt_30: "f32[544]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
    unsqueeze_1328: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(primals_635, 0);  primals_635 = None
    unsqueeze_1329: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 2);  unsqueeze_1328 = None
    unsqueeze_1330: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 3);  unsqueeze_1329 = None
    sum_62: "f32[544]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_151: "f32[4, 544, 7, 7]" = torch.ops.aten.sub.Tensor(cat_42, unsqueeze_1330);  cat_42 = unsqueeze_1330 = None
    mul_603: "f32[4, 544, 7, 7]" = torch.ops.aten.mul.Tensor(where_30, sub_151);  sub_151 = None
    sum_63: "f32[544]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 2, 3]);  mul_603 = None
    mul_608: "f32[544]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_271);  primals_271 = None
    unsqueeze_1337: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_1338: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    mul_609: "f32[4, 544, 7, 7]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_1339);  where_30 = unsqueeze_1339 = None
    mul_610: "f32[544]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_151: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_609, 1, 0, 512)
    slice_152: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_609, 1, 512, 544);  mul_609 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_406: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_401, slice_151);  add_401 = slice_151 = None
    add_407: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_402, slice_152);  add_402 = slice_152 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(add_407, relu_89, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_407 = primals_270 = None
    getitem_92: "f32[4, 128, 7, 7]" = convolution_backward_30[0]
    getitem_93: "f32[32, 128, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    le_31: "b8[4, 128, 7, 7]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
    where_31: "f32[4, 128, 7, 7]" = torch.ops.aten.where.self(le_31, full_default, getitem_92);  le_31 = getitem_92 = None
    add_408: "f32[128]" = torch.ops.aten.add.Tensor(primals_633, 1e-05);  primals_633 = None
    rsqrt_31: "f32[128]" = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
    unsqueeze_1340: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_632, 0);  primals_632 = None
    unsqueeze_1341: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
    unsqueeze_1342: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None
    sum_64: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_152: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_1342);  convolution_88 = unsqueeze_1342 = None
    mul_611: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_31, sub_152);  sub_152 = None
    sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_616: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_268);  primals_268 = None
    unsqueeze_1349: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_1350: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
    unsqueeze_1351: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
    mul_617: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_1351);  where_31 = unsqueeze_1351 = None
    mul_618: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_617, relu_88, primals_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_617 = primals_267 = None
    getitem_95: "f32[4, 512, 7, 7]" = convolution_backward_31[0]
    getitem_96: "f32[128, 512, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    le_32: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
    where_32: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_32, full_default, getitem_95);  le_32 = getitem_95 = None
    add_409: "f32[512]" = torch.ops.aten.add.Tensor(primals_630, 1e-05);  primals_630 = None
    rsqrt_32: "f32[512]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
    unsqueeze_1352: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_629, 0);  primals_629 = None
    unsqueeze_1353: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 2);  unsqueeze_1352 = None
    unsqueeze_1354: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 3);  unsqueeze_1353 = None
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_153: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(avg_pool2d_2, unsqueeze_1354);  avg_pool2d_2 = unsqueeze_1354 = None
    mul_619: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_32, sub_153);  sub_153 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
    mul_624: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_265);  primals_265 = None
    unsqueeze_1361: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_1362: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
    unsqueeze_1363: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
    mul_625: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_1363);  where_32 = unsqueeze_1363 = None
    mul_626: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_153: "f32[4, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 0, 512);  mul_625 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_410: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_406, slice_153);  add_406 = slice_153 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    avg_pool2d_backward: "f32[4, 512, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(add_410, convolution_87, [2, 2], [2, 2], [0, 0], False, True, None);  add_410 = convolution_87 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward, relu_87, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward = primals_264 = None
    getitem_98: "f32[4, 1024, 14, 14]" = convolution_backward_32[0]
    getitem_99: "f32[512, 1024, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    le_33: "b8[4, 1024, 14, 14]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
    where_33: "f32[4, 1024, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, getitem_98);  le_33 = getitem_98 = None
    add_411: "f32[1024]" = torch.ops.aten.add.Tensor(primals_627, 1e-05);  primals_627 = None
    rsqrt_33: "f32[1024]" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
    unsqueeze_1364: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_626, 0);  primals_626 = None
    unsqueeze_1365: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 2);  unsqueeze_1364 = None
    unsqueeze_1366: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 3);  unsqueeze_1365 = None
    sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_154: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_1366);  cat_41 = unsqueeze_1366 = None
    mul_627: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_154);  sub_154 = None
    sum_69: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_632: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_262);  primals_262 = None
    unsqueeze_1373: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_1374: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
    unsqueeze_1375: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
    mul_633: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_1375);  where_33 = unsqueeze_1375 = None
    mul_634: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    slice_154: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 0, 256)
    slice_155: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 256, 288)
    slice_156: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 288, 320)
    slice_157: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 320, 352)
    slice_158: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 352, 384)
    slice_159: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 384, 416)
    slice_160: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 416, 448)
    slice_161: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 448, 480)
    slice_162: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 480, 512)
    slice_163: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 512, 544)
    slice_164: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 544, 576)
    slice_165: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 576, 608)
    slice_166: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 608, 640)
    slice_167: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 640, 672)
    slice_168: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 672, 704)
    slice_169: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 704, 736)
    slice_170: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 736, 768)
    slice_171: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 768, 800)
    slice_172: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 800, 832)
    slice_173: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 832, 864)
    slice_174: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 864, 896)
    slice_175: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 896, 928)
    slice_176: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 928, 960)
    slice_177: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 960, 992)
    slice_178: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_633, 1, 992, 1024);  mul_633 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_178, relu_86, primals_261, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_178 = primals_261 = None
    getitem_101: "f32[4, 128, 14, 14]" = convolution_backward_33[0]
    getitem_102: "f32[32, 128, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    le_34: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
    where_34: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, getitem_101);  le_34 = getitem_101 = None
    add_412: "f32[128]" = torch.ops.aten.add.Tensor(primals_624, 1e-05);  primals_624 = None
    rsqrt_34: "f32[128]" = torch.ops.aten.rsqrt.default(add_412);  add_412 = None
    unsqueeze_1376: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_623, 0);  primals_623 = None
    unsqueeze_1377: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 2);  unsqueeze_1376 = None
    unsqueeze_1378: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 3);  unsqueeze_1377 = None
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_155: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_1378);  convolution_85 = unsqueeze_1378 = None
    mul_635: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_155);  sub_155 = None
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
    mul_640: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_259);  primals_259 = None
    unsqueeze_1385: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_1386: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
    unsqueeze_1387: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
    mul_641: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_1387);  where_34 = unsqueeze_1387 = None
    mul_642: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_641, relu_85, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_641 = primals_258 = None
    getitem_104: "f32[4, 992, 14, 14]" = convolution_backward_34[0]
    getitem_105: "f32[128, 992, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    le_35: "b8[4, 992, 14, 14]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
    where_35: "f32[4, 992, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_104);  le_35 = getitem_104 = None
    add_413: "f32[992]" = torch.ops.aten.add.Tensor(primals_621, 1e-05);  primals_621 = None
    rsqrt_35: "f32[992]" = torch.ops.aten.rsqrt.default(add_413);  add_413 = None
    unsqueeze_1388: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(primals_620, 0);  primals_620 = None
    unsqueeze_1389: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 2);  unsqueeze_1388 = None
    unsqueeze_1390: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 3);  unsqueeze_1389 = None
    sum_72: "f32[992]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_156: "f32[4, 992, 14, 14]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_1390);  cat_40 = unsqueeze_1390 = None
    mul_643: "f32[4, 992, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_156);  sub_156 = None
    sum_73: "f32[992]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
    mul_648: "f32[992]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_256);  primals_256 = None
    unsqueeze_1397: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_1398: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    mul_649: "f32[4, 992, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_1399);  where_35 = unsqueeze_1399 = None
    mul_650: "f32[992]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_179: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 0, 256)
    slice_180: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 256, 288)
    slice_181: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 288, 320)
    slice_182: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 320, 352)
    slice_183: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 352, 384)
    slice_184: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 384, 416)
    slice_185: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 416, 448)
    slice_186: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 448, 480)
    slice_187: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 480, 512)
    slice_188: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 512, 544)
    slice_189: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 544, 576)
    slice_190: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 576, 608)
    slice_191: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 608, 640)
    slice_192: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 640, 672)
    slice_193: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 672, 704)
    slice_194: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 704, 736)
    slice_195: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 736, 768)
    slice_196: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 768, 800)
    slice_197: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 800, 832)
    slice_198: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 832, 864)
    slice_199: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 864, 896)
    slice_200: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 896, 928)
    slice_201: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 928, 960)
    slice_202: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_649, 1, 960, 992);  mul_649 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_414: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(slice_154, slice_179);  slice_154 = slice_179 = None
    add_415: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_155, slice_180);  slice_155 = slice_180 = None
    add_416: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_156, slice_181);  slice_156 = slice_181 = None
    add_417: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_157, slice_182);  slice_157 = slice_182 = None
    add_418: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_158, slice_183);  slice_158 = slice_183 = None
    add_419: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_159, slice_184);  slice_159 = slice_184 = None
    add_420: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_160, slice_185);  slice_160 = slice_185 = None
    add_421: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_161, slice_186);  slice_161 = slice_186 = None
    add_422: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_162, slice_187);  slice_162 = slice_187 = None
    add_423: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_163, slice_188);  slice_163 = slice_188 = None
    add_424: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_164, slice_189);  slice_164 = slice_189 = None
    add_425: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_165, slice_190);  slice_165 = slice_190 = None
    add_426: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_166, slice_191);  slice_166 = slice_191 = None
    add_427: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_167, slice_192);  slice_167 = slice_192 = None
    add_428: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_168, slice_193);  slice_168 = slice_193 = None
    add_429: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_169, slice_194);  slice_169 = slice_194 = None
    add_430: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_170, slice_195);  slice_170 = slice_195 = None
    add_431: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_171, slice_196);  slice_171 = slice_196 = None
    add_432: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_172, slice_197);  slice_172 = slice_197 = None
    add_433: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_173, slice_198);  slice_173 = slice_198 = None
    add_434: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_174, slice_199);  slice_174 = slice_199 = None
    add_435: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_175, slice_200);  slice_175 = slice_200 = None
    add_436: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_176, slice_201);  slice_176 = slice_201 = None
    add_437: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_177, slice_202);  slice_177 = slice_202 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_437, relu_84, primals_255, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_437 = primals_255 = None
    getitem_107: "f32[4, 128, 14, 14]" = convolution_backward_35[0]
    getitem_108: "f32[32, 128, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    le_36: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
    where_36: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, getitem_107);  le_36 = getitem_107 = None
    add_438: "f32[128]" = torch.ops.aten.add.Tensor(primals_618, 1e-05);  primals_618 = None
    rsqrt_36: "f32[128]" = torch.ops.aten.rsqrt.default(add_438);  add_438 = None
    unsqueeze_1400: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_617, 0);  primals_617 = None
    unsqueeze_1401: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
    unsqueeze_1402: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_157: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_1402);  convolution_83 = unsqueeze_1402 = None
    mul_651: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_157);  sub_157 = None
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_656: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_253);  primals_253 = None
    unsqueeze_1409: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_1410: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    mul_657: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_1411);  where_36 = unsqueeze_1411 = None
    mul_658: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_657, relu_83, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_657 = primals_252 = None
    getitem_110: "f32[4, 960, 14, 14]" = convolution_backward_36[0]
    getitem_111: "f32[128, 960, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    le_37: "b8[4, 960, 14, 14]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
    where_37: "f32[4, 960, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, getitem_110);  le_37 = getitem_110 = None
    add_439: "f32[960]" = torch.ops.aten.add.Tensor(primals_615, 1e-05);  primals_615 = None
    rsqrt_37: "f32[960]" = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
    unsqueeze_1412: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_614, 0);  primals_614 = None
    unsqueeze_1413: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
    unsqueeze_1414: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None
    sum_76: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_158: "f32[4, 960, 14, 14]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_1414);  cat_39 = unsqueeze_1414 = None
    mul_659: "f32[4, 960, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_158);  sub_158 = None
    sum_77: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
    mul_664: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_250);  primals_250 = None
    unsqueeze_1421: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_1422: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    mul_665: "f32[4, 960, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_1423);  where_37 = unsqueeze_1423 = None
    mul_666: "f32[960]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_203: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 0, 256)
    slice_204: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 256, 288)
    slice_205: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 288, 320)
    slice_206: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 320, 352)
    slice_207: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 352, 384)
    slice_208: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 384, 416)
    slice_209: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 416, 448)
    slice_210: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 448, 480)
    slice_211: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 480, 512)
    slice_212: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 512, 544)
    slice_213: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 544, 576)
    slice_214: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 576, 608)
    slice_215: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 608, 640)
    slice_216: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 640, 672)
    slice_217: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 672, 704)
    slice_218: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 704, 736)
    slice_219: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 736, 768)
    slice_220: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 768, 800)
    slice_221: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 800, 832)
    slice_222: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 832, 864)
    slice_223: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 864, 896)
    slice_224: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 896, 928)
    slice_225: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 928, 960);  mul_665 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_440: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_414, slice_203);  add_414 = slice_203 = None
    add_441: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_415, slice_204);  add_415 = slice_204 = None
    add_442: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_416, slice_205);  add_416 = slice_205 = None
    add_443: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_417, slice_206);  add_417 = slice_206 = None
    add_444: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_418, slice_207);  add_418 = slice_207 = None
    add_445: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_419, slice_208);  add_419 = slice_208 = None
    add_446: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_420, slice_209);  add_420 = slice_209 = None
    add_447: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_421, slice_210);  add_421 = slice_210 = None
    add_448: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_422, slice_211);  add_422 = slice_211 = None
    add_449: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_423, slice_212);  add_423 = slice_212 = None
    add_450: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_424, slice_213);  add_424 = slice_213 = None
    add_451: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_425, slice_214);  add_425 = slice_214 = None
    add_452: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_426, slice_215);  add_426 = slice_215 = None
    add_453: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_427, slice_216);  add_427 = slice_216 = None
    add_454: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_428, slice_217);  add_428 = slice_217 = None
    add_455: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_429, slice_218);  add_429 = slice_218 = None
    add_456: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_430, slice_219);  add_430 = slice_219 = None
    add_457: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_431, slice_220);  add_431 = slice_220 = None
    add_458: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_432, slice_221);  add_432 = slice_221 = None
    add_459: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_433, slice_222);  add_433 = slice_222 = None
    add_460: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_434, slice_223);  add_434 = slice_223 = None
    add_461: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_435, slice_224);  add_435 = slice_224 = None
    add_462: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_436, slice_225);  add_436 = slice_225 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(add_462, relu_82, primals_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_462 = primals_249 = None
    getitem_113: "f32[4, 128, 14, 14]" = convolution_backward_37[0]
    getitem_114: "f32[32, 128, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    le_38: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
    where_38: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, getitem_113);  le_38 = getitem_113 = None
    add_463: "f32[128]" = torch.ops.aten.add.Tensor(primals_612, 1e-05);  primals_612 = None
    rsqrt_38: "f32[128]" = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
    unsqueeze_1424: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_611, 0);  primals_611 = None
    unsqueeze_1425: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
    unsqueeze_1426: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None
    sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_159: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_1426);  convolution_81 = unsqueeze_1426 = None
    mul_667: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_159);  sub_159 = None
    sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_672: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_247);  primals_247 = None
    unsqueeze_1433: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_1434: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
    unsqueeze_1435: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
    mul_673: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_1435);  where_38 = unsqueeze_1435 = None
    mul_674: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_673, relu_81, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_673 = primals_246 = None
    getitem_116: "f32[4, 928, 14, 14]" = convolution_backward_38[0]
    getitem_117: "f32[128, 928, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    le_39: "b8[4, 928, 14, 14]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
    where_39: "f32[4, 928, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, getitem_116);  le_39 = getitem_116 = None
    add_464: "f32[928]" = torch.ops.aten.add.Tensor(primals_609, 1e-05);  primals_609 = None
    rsqrt_39: "f32[928]" = torch.ops.aten.rsqrt.default(add_464);  add_464 = None
    unsqueeze_1436: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(primals_608, 0);  primals_608 = None
    unsqueeze_1437: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 2);  unsqueeze_1436 = None
    unsqueeze_1438: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 3);  unsqueeze_1437 = None
    sum_80: "f32[928]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_160: "f32[4, 928, 14, 14]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_1438);  cat_38 = unsqueeze_1438 = None
    mul_675: "f32[4, 928, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_160);  sub_160 = None
    sum_81: "f32[928]" = torch.ops.aten.sum.dim_IntList(mul_675, [0, 2, 3]);  mul_675 = None
    mul_680: "f32[928]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_244);  primals_244 = None
    unsqueeze_1445: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_1446: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
    unsqueeze_1447: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
    mul_681: "f32[4, 928, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_1447);  where_39 = unsqueeze_1447 = None
    mul_682: "f32[928]" = torch.ops.aten.mul.Tensor(sum_81, rsqrt_39);  sum_81 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_226: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 0, 256)
    slice_227: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 256, 288)
    slice_228: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 288, 320)
    slice_229: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 320, 352)
    slice_230: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 352, 384)
    slice_231: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 384, 416)
    slice_232: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 416, 448)
    slice_233: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 448, 480)
    slice_234: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 480, 512)
    slice_235: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 512, 544)
    slice_236: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 544, 576)
    slice_237: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 576, 608)
    slice_238: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 608, 640)
    slice_239: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 640, 672)
    slice_240: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 672, 704)
    slice_241: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 704, 736)
    slice_242: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 736, 768)
    slice_243: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 768, 800)
    slice_244: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 800, 832)
    slice_245: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 832, 864)
    slice_246: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 864, 896)
    slice_247: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_681, 1, 896, 928);  mul_681 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_465: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_440, slice_226);  add_440 = slice_226 = None
    add_466: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_441, slice_227);  add_441 = slice_227 = None
    add_467: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_442, slice_228);  add_442 = slice_228 = None
    add_468: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_443, slice_229);  add_443 = slice_229 = None
    add_469: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_444, slice_230);  add_444 = slice_230 = None
    add_470: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_445, slice_231);  add_445 = slice_231 = None
    add_471: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_446, slice_232);  add_446 = slice_232 = None
    add_472: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_447, slice_233);  add_447 = slice_233 = None
    add_473: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_448, slice_234);  add_448 = slice_234 = None
    add_474: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_449, slice_235);  add_449 = slice_235 = None
    add_475: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_450, slice_236);  add_450 = slice_236 = None
    add_476: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_451, slice_237);  add_451 = slice_237 = None
    add_477: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_452, slice_238);  add_452 = slice_238 = None
    add_478: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_453, slice_239);  add_453 = slice_239 = None
    add_479: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_454, slice_240);  add_454 = slice_240 = None
    add_480: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_455, slice_241);  add_455 = slice_241 = None
    add_481: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_456, slice_242);  add_456 = slice_242 = None
    add_482: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_457, slice_243);  add_457 = slice_243 = None
    add_483: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_458, slice_244);  add_458 = slice_244 = None
    add_484: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_459, slice_245);  add_459 = slice_245 = None
    add_485: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_460, slice_246);  add_460 = slice_246 = None
    add_486: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_461, slice_247);  add_461 = slice_247 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(add_486, relu_80, primals_243, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_486 = primals_243 = None
    getitem_119: "f32[4, 128, 14, 14]" = convolution_backward_39[0]
    getitem_120: "f32[32, 128, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    le_40: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
    where_40: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, getitem_119);  le_40 = getitem_119 = None
    add_487: "f32[128]" = torch.ops.aten.add.Tensor(primals_606, 1e-05);  primals_606 = None
    rsqrt_40: "f32[128]" = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
    unsqueeze_1448: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_605, 0);  primals_605 = None
    unsqueeze_1449: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 2);  unsqueeze_1448 = None
    unsqueeze_1450: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 3);  unsqueeze_1449 = None
    sum_82: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_161: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_1450);  convolution_79 = unsqueeze_1450 = None
    mul_683: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_161);  sub_161 = None
    sum_83: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3]);  mul_683 = None
    mul_688: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_241);  primals_241 = None
    unsqueeze_1457: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_1458: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
    unsqueeze_1459: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
    mul_689: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, unsqueeze_1459);  where_40 = unsqueeze_1459 = None
    mul_690: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, rsqrt_40);  sum_83 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_689, relu_79, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_689 = primals_240 = None
    getitem_122: "f32[4, 896, 14, 14]" = convolution_backward_40[0]
    getitem_123: "f32[128, 896, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    le_41: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
    where_41: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, getitem_122);  le_41 = getitem_122 = None
    add_488: "f32[896]" = torch.ops.aten.add.Tensor(primals_603, 1e-05);  primals_603 = None
    rsqrt_41: "f32[896]" = torch.ops.aten.rsqrt.default(add_488);  add_488 = None
    unsqueeze_1460: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_602, 0);  primals_602 = None
    unsqueeze_1461: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 2);  unsqueeze_1460 = None
    unsqueeze_1462: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 3);  unsqueeze_1461 = None
    sum_84: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_162: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_1462);  cat_37 = unsqueeze_1462 = None
    mul_691: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_162);  sub_162 = None
    sum_85: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_696: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_238);  primals_238 = None
    unsqueeze_1469: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_1470: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
    unsqueeze_1471: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
    mul_697: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, unsqueeze_1471);  where_41 = unsqueeze_1471 = None
    mul_698: "f32[896]" = torch.ops.aten.mul.Tensor(sum_85, rsqrt_41);  sum_85 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_248: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 0, 256)
    slice_249: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 256, 288)
    slice_250: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 288, 320)
    slice_251: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 320, 352)
    slice_252: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 352, 384)
    slice_253: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 384, 416)
    slice_254: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 416, 448)
    slice_255: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 448, 480)
    slice_256: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 480, 512)
    slice_257: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 512, 544)
    slice_258: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 544, 576)
    slice_259: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 576, 608)
    slice_260: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 608, 640)
    slice_261: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 640, 672)
    slice_262: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 672, 704)
    slice_263: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 704, 736)
    slice_264: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 736, 768)
    slice_265: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 768, 800)
    slice_266: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 800, 832)
    slice_267: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 832, 864)
    slice_268: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_697, 1, 864, 896);  mul_697 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_489: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_465, slice_248);  add_465 = slice_248 = None
    add_490: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_466, slice_249);  add_466 = slice_249 = None
    add_491: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_467, slice_250);  add_467 = slice_250 = None
    add_492: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_468, slice_251);  add_468 = slice_251 = None
    add_493: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_469, slice_252);  add_469 = slice_252 = None
    add_494: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_470, slice_253);  add_470 = slice_253 = None
    add_495: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_471, slice_254);  add_471 = slice_254 = None
    add_496: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_472, slice_255);  add_472 = slice_255 = None
    add_497: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_473, slice_256);  add_473 = slice_256 = None
    add_498: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_474, slice_257);  add_474 = slice_257 = None
    add_499: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_475, slice_258);  add_475 = slice_258 = None
    add_500: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_476, slice_259);  add_476 = slice_259 = None
    add_501: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_477, slice_260);  add_477 = slice_260 = None
    add_502: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_478, slice_261);  add_478 = slice_261 = None
    add_503: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_479, slice_262);  add_479 = slice_262 = None
    add_504: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_480, slice_263);  add_480 = slice_263 = None
    add_505: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_481, slice_264);  add_481 = slice_264 = None
    add_506: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_482, slice_265);  add_482 = slice_265 = None
    add_507: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_483, slice_266);  add_483 = slice_266 = None
    add_508: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_484, slice_267);  add_484 = slice_267 = None
    add_509: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_485, slice_268);  add_485 = slice_268 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(add_509, relu_78, primals_237, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_509 = primals_237 = None
    getitem_125: "f32[4, 128, 14, 14]" = convolution_backward_41[0]
    getitem_126: "f32[32, 128, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    le_42: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
    where_42: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, getitem_125);  le_42 = getitem_125 = None
    add_510: "f32[128]" = torch.ops.aten.add.Tensor(primals_600, 1e-05);  primals_600 = None
    rsqrt_42: "f32[128]" = torch.ops.aten.rsqrt.default(add_510);  add_510 = None
    unsqueeze_1472: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_599, 0);  primals_599 = None
    unsqueeze_1473: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 2);  unsqueeze_1472 = None
    unsqueeze_1474: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 3);  unsqueeze_1473 = None
    sum_86: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_163: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_1474);  convolution_77 = unsqueeze_1474 = None
    mul_699: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_163);  sub_163 = None
    sum_87: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 2, 3]);  mul_699 = None
    mul_704: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_235);  primals_235 = None
    unsqueeze_1481: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_1482: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    mul_705: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, unsqueeze_1483);  where_42 = unsqueeze_1483 = None
    mul_706: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, rsqrt_42);  sum_87 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_705, relu_77, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_705 = primals_234 = None
    getitem_128: "f32[4, 864, 14, 14]" = convolution_backward_42[0]
    getitem_129: "f32[128, 864, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    le_43: "b8[4, 864, 14, 14]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
    where_43: "f32[4, 864, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, getitem_128);  le_43 = getitem_128 = None
    add_511: "f32[864]" = torch.ops.aten.add.Tensor(primals_597, 1e-05);  primals_597 = None
    rsqrt_43: "f32[864]" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
    unsqueeze_1484: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(primals_596, 0);  primals_596 = None
    unsqueeze_1485: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
    unsqueeze_1486: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None
    sum_88: "f32[864]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_164: "f32[4, 864, 14, 14]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_1486);  cat_36 = unsqueeze_1486 = None
    mul_707: "f32[4, 864, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_164);  sub_164 = None
    sum_89: "f32[864]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_712: "f32[864]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_232);  primals_232 = None
    unsqueeze_1493: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_1494: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
    unsqueeze_1495: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
    mul_713: "f32[4, 864, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_1495);  where_43 = unsqueeze_1495 = None
    mul_714: "f32[864]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_43);  sum_89 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_269: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 0, 256)
    slice_270: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 256, 288)
    slice_271: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 288, 320)
    slice_272: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 320, 352)
    slice_273: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 352, 384)
    slice_274: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 384, 416)
    slice_275: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 416, 448)
    slice_276: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 448, 480)
    slice_277: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 480, 512)
    slice_278: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 512, 544)
    slice_279: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 544, 576)
    slice_280: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 576, 608)
    slice_281: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 608, 640)
    slice_282: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 640, 672)
    slice_283: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 672, 704)
    slice_284: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 704, 736)
    slice_285: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 736, 768)
    slice_286: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 768, 800)
    slice_287: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 800, 832)
    slice_288: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_713, 1, 832, 864);  mul_713 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_512: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_489, slice_269);  add_489 = slice_269 = None
    add_513: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_490, slice_270);  add_490 = slice_270 = None
    add_514: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_491, slice_271);  add_491 = slice_271 = None
    add_515: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_492, slice_272);  add_492 = slice_272 = None
    add_516: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_493, slice_273);  add_493 = slice_273 = None
    add_517: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_494, slice_274);  add_494 = slice_274 = None
    add_518: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_495, slice_275);  add_495 = slice_275 = None
    add_519: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_496, slice_276);  add_496 = slice_276 = None
    add_520: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_497, slice_277);  add_497 = slice_277 = None
    add_521: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_498, slice_278);  add_498 = slice_278 = None
    add_522: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_499, slice_279);  add_499 = slice_279 = None
    add_523: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_500, slice_280);  add_500 = slice_280 = None
    add_524: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_501, slice_281);  add_501 = slice_281 = None
    add_525: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_502, slice_282);  add_502 = slice_282 = None
    add_526: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_503, slice_283);  add_503 = slice_283 = None
    add_527: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_504, slice_284);  add_504 = slice_284 = None
    add_528: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_505, slice_285);  add_505 = slice_285 = None
    add_529: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_506, slice_286);  add_506 = slice_286 = None
    add_530: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_507, slice_287);  add_507 = slice_287 = None
    add_531: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_508, slice_288);  add_508 = slice_288 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(add_531, relu_76, primals_231, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_531 = primals_231 = None
    getitem_131: "f32[4, 128, 14, 14]" = convolution_backward_43[0]
    getitem_132: "f32[32, 128, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    le_44: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    where_44: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, getitem_131);  le_44 = getitem_131 = None
    add_532: "f32[128]" = torch.ops.aten.add.Tensor(primals_594, 1e-05);  primals_594 = None
    rsqrt_44: "f32[128]" = torch.ops.aten.rsqrt.default(add_532);  add_532 = None
    unsqueeze_1496: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_593, 0);  primals_593 = None
    unsqueeze_1497: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 2);  unsqueeze_1496 = None
    unsqueeze_1498: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 3);  unsqueeze_1497 = None
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_165: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_1498);  convolution_75 = unsqueeze_1498 = None
    mul_715: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_165);  sub_165 = None
    sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_720: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_229);  primals_229 = None
    unsqueeze_1505: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_1506: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
    unsqueeze_1507: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
    mul_721: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_1507);  where_44 = unsqueeze_1507 = None
    mul_722: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_44);  sum_91 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_721, relu_75, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_721 = primals_228 = None
    getitem_134: "f32[4, 832, 14, 14]" = convolution_backward_44[0]
    getitem_135: "f32[128, 832, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    le_45: "b8[4, 832, 14, 14]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_45: "f32[4, 832, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, getitem_134);  le_45 = getitem_134 = None
    add_533: "f32[832]" = torch.ops.aten.add.Tensor(primals_591, 1e-05);  primals_591 = None
    rsqrt_45: "f32[832]" = torch.ops.aten.rsqrt.default(add_533);  add_533 = None
    unsqueeze_1508: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(primals_590, 0);  primals_590 = None
    unsqueeze_1509: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 2);  unsqueeze_1508 = None
    unsqueeze_1510: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 3);  unsqueeze_1509 = None
    sum_92: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_166: "f32[4, 832, 14, 14]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_1510);  cat_35 = unsqueeze_1510 = None
    mul_723: "f32[4, 832, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_166);  sub_166 = None
    sum_93: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_723, [0, 2, 3]);  mul_723 = None
    mul_728: "f32[832]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_226);  primals_226 = None
    unsqueeze_1517: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_1518: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
    unsqueeze_1519: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
    mul_729: "f32[4, 832, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_1519);  where_45 = unsqueeze_1519 = None
    mul_730: "f32[832]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_45);  sum_93 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_289: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 0, 256)
    slice_290: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 256, 288)
    slice_291: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 288, 320)
    slice_292: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 320, 352)
    slice_293: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 352, 384)
    slice_294: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 384, 416)
    slice_295: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 416, 448)
    slice_296: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 448, 480)
    slice_297: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 480, 512)
    slice_298: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 512, 544)
    slice_299: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 544, 576)
    slice_300: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 576, 608)
    slice_301: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 608, 640)
    slice_302: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 640, 672)
    slice_303: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 672, 704)
    slice_304: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 704, 736)
    slice_305: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 736, 768)
    slice_306: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 768, 800)
    slice_307: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_729, 1, 800, 832);  mul_729 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_534: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_512, slice_289);  add_512 = slice_289 = None
    add_535: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_513, slice_290);  add_513 = slice_290 = None
    add_536: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_514, slice_291);  add_514 = slice_291 = None
    add_537: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_515, slice_292);  add_515 = slice_292 = None
    add_538: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_516, slice_293);  add_516 = slice_293 = None
    add_539: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_517, slice_294);  add_517 = slice_294 = None
    add_540: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_518, slice_295);  add_518 = slice_295 = None
    add_541: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_519, slice_296);  add_519 = slice_296 = None
    add_542: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_520, slice_297);  add_520 = slice_297 = None
    add_543: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_521, slice_298);  add_521 = slice_298 = None
    add_544: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_522, slice_299);  add_522 = slice_299 = None
    add_545: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_523, slice_300);  add_523 = slice_300 = None
    add_546: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_524, slice_301);  add_524 = slice_301 = None
    add_547: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_525, slice_302);  add_525 = slice_302 = None
    add_548: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_526, slice_303);  add_526 = slice_303 = None
    add_549: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_527, slice_304);  add_527 = slice_304 = None
    add_550: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_528, slice_305);  add_528 = slice_305 = None
    add_551: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_529, slice_306);  add_529 = slice_306 = None
    add_552: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_530, slice_307);  add_530 = slice_307 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(add_552, relu_74, primals_225, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_552 = primals_225 = None
    getitem_137: "f32[4, 128, 14, 14]" = convolution_backward_45[0]
    getitem_138: "f32[32, 128, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    le_46: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
    where_46: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, getitem_137);  le_46 = getitem_137 = None
    add_553: "f32[128]" = torch.ops.aten.add.Tensor(primals_588, 1e-05);  primals_588 = None
    rsqrt_46: "f32[128]" = torch.ops.aten.rsqrt.default(add_553);  add_553 = None
    unsqueeze_1520: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_587, 0);  primals_587 = None
    unsqueeze_1521: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 2);  unsqueeze_1520 = None
    unsqueeze_1522: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 3);  unsqueeze_1521 = None
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_167: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_1522);  convolution_73 = unsqueeze_1522 = None
    mul_731: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_167);  sub_167 = None
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3]);  mul_731 = None
    mul_736: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_223);  primals_223 = None
    unsqueeze_1529: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_1530: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
    unsqueeze_1531: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
    mul_737: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, unsqueeze_1531);  where_46 = unsqueeze_1531 = None
    mul_738: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_46);  sum_95 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_737, relu_73, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_737 = primals_222 = None
    getitem_140: "f32[4, 800, 14, 14]" = convolution_backward_46[0]
    getitem_141: "f32[128, 800, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    le_47: "b8[4, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
    where_47: "f32[4, 800, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, getitem_140);  le_47 = getitem_140 = None
    add_554: "f32[800]" = torch.ops.aten.add.Tensor(primals_585, 1e-05);  primals_585 = None
    rsqrt_47: "f32[800]" = torch.ops.aten.rsqrt.default(add_554);  add_554 = None
    unsqueeze_1532: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(primals_584, 0);  primals_584 = None
    unsqueeze_1533: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 2);  unsqueeze_1532 = None
    unsqueeze_1534: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 3);  unsqueeze_1533 = None
    sum_96: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_168: "f32[4, 800, 14, 14]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_1534);  cat_34 = unsqueeze_1534 = None
    mul_739: "f32[4, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_168);  sub_168 = None
    sum_97: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
    mul_744: "f32[800]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_220);  primals_220 = None
    unsqueeze_1541: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_1542: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
    unsqueeze_1543: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
    mul_745: "f32[4, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_1543);  where_47 = unsqueeze_1543 = None
    mul_746: "f32[800]" = torch.ops.aten.mul.Tensor(sum_97, rsqrt_47);  sum_97 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_308: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 0, 256)
    slice_309: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 256, 288)
    slice_310: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 288, 320)
    slice_311: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 320, 352)
    slice_312: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 352, 384)
    slice_313: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 384, 416)
    slice_314: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 416, 448)
    slice_315: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 448, 480)
    slice_316: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 480, 512)
    slice_317: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 512, 544)
    slice_318: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 544, 576)
    slice_319: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 576, 608)
    slice_320: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 608, 640)
    slice_321: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 640, 672)
    slice_322: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 672, 704)
    slice_323: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 704, 736)
    slice_324: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 736, 768)
    slice_325: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 768, 800);  mul_745 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_555: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_534, slice_308);  add_534 = slice_308 = None
    add_556: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_535, slice_309);  add_535 = slice_309 = None
    add_557: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_536, slice_310);  add_536 = slice_310 = None
    add_558: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_537, slice_311);  add_537 = slice_311 = None
    add_559: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_538, slice_312);  add_538 = slice_312 = None
    add_560: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_539, slice_313);  add_539 = slice_313 = None
    add_561: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_540, slice_314);  add_540 = slice_314 = None
    add_562: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_541, slice_315);  add_541 = slice_315 = None
    add_563: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_542, slice_316);  add_542 = slice_316 = None
    add_564: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_543, slice_317);  add_543 = slice_317 = None
    add_565: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_544, slice_318);  add_544 = slice_318 = None
    add_566: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_545, slice_319);  add_545 = slice_319 = None
    add_567: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_546, slice_320);  add_546 = slice_320 = None
    add_568: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_547, slice_321);  add_547 = slice_321 = None
    add_569: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_548, slice_322);  add_548 = slice_322 = None
    add_570: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_549, slice_323);  add_549 = slice_323 = None
    add_571: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_550, slice_324);  add_550 = slice_324 = None
    add_572: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_551, slice_325);  add_551 = slice_325 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(add_572, relu_72, primals_219, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_572 = primals_219 = None
    getitem_143: "f32[4, 128, 14, 14]" = convolution_backward_47[0]
    getitem_144: "f32[32, 128, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    le_48: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
    where_48: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, getitem_143);  le_48 = getitem_143 = None
    add_573: "f32[128]" = torch.ops.aten.add.Tensor(primals_582, 1e-05);  primals_582 = None
    rsqrt_48: "f32[128]" = torch.ops.aten.rsqrt.default(add_573);  add_573 = None
    unsqueeze_1544: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_581, 0);  primals_581 = None
    unsqueeze_1545: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
    unsqueeze_1546: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None
    sum_98: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_169: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_1546);  convolution_71 = unsqueeze_1546 = None
    mul_747: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_169);  sub_169 = None
    sum_99: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 2, 3]);  mul_747 = None
    mul_752: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_217);  primals_217 = None
    unsqueeze_1553: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_1554: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    mul_753: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, unsqueeze_1555);  where_48 = unsqueeze_1555 = None
    mul_754: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, rsqrt_48);  sum_99 = rsqrt_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_753, relu_71, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_753 = primals_216 = None
    getitem_146: "f32[4, 768, 14, 14]" = convolution_backward_48[0]
    getitem_147: "f32[128, 768, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    le_49: "b8[4, 768, 14, 14]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
    where_49: "f32[4, 768, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, getitem_146);  le_49 = getitem_146 = None
    add_574: "f32[768]" = torch.ops.aten.add.Tensor(primals_579, 1e-05);  primals_579 = None
    rsqrt_49: "f32[768]" = torch.ops.aten.rsqrt.default(add_574);  add_574 = None
    unsqueeze_1556: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_578, 0);  primals_578 = None
    unsqueeze_1557: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
    unsqueeze_1558: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_170: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_1558);  cat_33 = unsqueeze_1558 = None
    mul_755: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_170);  sub_170 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 2, 3]);  mul_755 = None
    mul_760: "f32[768]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_214);  primals_214 = None
    unsqueeze_1565: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_1566: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    mul_761: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, unsqueeze_1567);  where_49 = unsqueeze_1567 = None
    mul_762: "f32[768]" = torch.ops.aten.mul.Tensor(sum_101, rsqrt_49);  sum_101 = rsqrt_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_326: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 0, 256)
    slice_327: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 256, 288)
    slice_328: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 288, 320)
    slice_329: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 320, 352)
    slice_330: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 352, 384)
    slice_331: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 384, 416)
    slice_332: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 416, 448)
    slice_333: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 448, 480)
    slice_334: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 480, 512)
    slice_335: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 512, 544)
    slice_336: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 544, 576)
    slice_337: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 576, 608)
    slice_338: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 608, 640)
    slice_339: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 640, 672)
    slice_340: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 672, 704)
    slice_341: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 704, 736)
    slice_342: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_761, 1, 736, 768);  mul_761 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_575: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_555, slice_326);  add_555 = slice_326 = None
    add_576: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_556, slice_327);  add_556 = slice_327 = None
    add_577: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_557, slice_328);  add_557 = slice_328 = None
    add_578: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_558, slice_329);  add_558 = slice_329 = None
    add_579: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_559, slice_330);  add_559 = slice_330 = None
    add_580: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_560, slice_331);  add_560 = slice_331 = None
    add_581: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_561, slice_332);  add_561 = slice_332 = None
    add_582: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_562, slice_333);  add_562 = slice_333 = None
    add_583: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_563, slice_334);  add_563 = slice_334 = None
    add_584: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_564, slice_335);  add_564 = slice_335 = None
    add_585: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_565, slice_336);  add_565 = slice_336 = None
    add_586: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_566, slice_337);  add_566 = slice_337 = None
    add_587: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_567, slice_338);  add_567 = slice_338 = None
    add_588: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_568, slice_339);  add_568 = slice_339 = None
    add_589: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_569, slice_340);  add_569 = slice_340 = None
    add_590: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_570, slice_341);  add_570 = slice_341 = None
    add_591: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_571, slice_342);  add_571 = slice_342 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_591, relu_70, primals_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_591 = primals_213 = None
    getitem_149: "f32[4, 128, 14, 14]" = convolution_backward_49[0]
    getitem_150: "f32[32, 128, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    le_50: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_50: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, getitem_149);  le_50 = getitem_149 = None
    add_592: "f32[128]" = torch.ops.aten.add.Tensor(primals_576, 1e-05);  primals_576 = None
    rsqrt_50: "f32[128]" = torch.ops.aten.rsqrt.default(add_592);  add_592 = None
    unsqueeze_1568: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_575, 0);  primals_575 = None
    unsqueeze_1569: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
    unsqueeze_1570: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None
    sum_102: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_171: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_1570);  convolution_69 = unsqueeze_1570 = None
    mul_763: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_171);  sub_171 = None
    sum_103: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 2, 3]);  mul_763 = None
    mul_768: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_211);  primals_211 = None
    unsqueeze_1577: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_1578: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    mul_769: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, unsqueeze_1579);  where_50 = unsqueeze_1579 = None
    mul_770: "f32[128]" = torch.ops.aten.mul.Tensor(sum_103, rsqrt_50);  sum_103 = rsqrt_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_769, relu_69, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_769 = primals_210 = None
    getitem_152: "f32[4, 736, 14, 14]" = convolution_backward_50[0]
    getitem_153: "f32[128, 736, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    le_51: "b8[4, 736, 14, 14]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
    where_51: "f32[4, 736, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, getitem_152);  le_51 = getitem_152 = None
    add_593: "f32[736]" = torch.ops.aten.add.Tensor(primals_573, 1e-05);  primals_573 = None
    rsqrt_51: "f32[736]" = torch.ops.aten.rsqrt.default(add_593);  add_593 = None
    unsqueeze_1580: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(primals_572, 0);  primals_572 = None
    unsqueeze_1581: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 2);  unsqueeze_1580 = None
    unsqueeze_1582: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 3);  unsqueeze_1581 = None
    sum_104: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_172: "f32[4, 736, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_1582);  cat_32 = unsqueeze_1582 = None
    mul_771: "f32[4, 736, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_172);  sub_172 = None
    sum_105: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_776: "f32[736]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_208);  primals_208 = None
    unsqueeze_1589: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_1590: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
    unsqueeze_1591: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
    mul_777: "f32[4, 736, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, unsqueeze_1591);  where_51 = unsqueeze_1591 = None
    mul_778: "f32[736]" = torch.ops.aten.mul.Tensor(sum_105, rsqrt_51);  sum_105 = rsqrt_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_343: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 0, 256)
    slice_344: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 256, 288)
    slice_345: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 288, 320)
    slice_346: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 320, 352)
    slice_347: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 352, 384)
    slice_348: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 384, 416)
    slice_349: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 416, 448)
    slice_350: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 448, 480)
    slice_351: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 480, 512)
    slice_352: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 512, 544)
    slice_353: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 544, 576)
    slice_354: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 576, 608)
    slice_355: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 608, 640)
    slice_356: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 640, 672)
    slice_357: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 672, 704)
    slice_358: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_777, 1, 704, 736);  mul_777 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_594: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_575, slice_343);  add_575 = slice_343 = None
    add_595: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_576, slice_344);  add_576 = slice_344 = None
    add_596: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_577, slice_345);  add_577 = slice_345 = None
    add_597: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_578, slice_346);  add_578 = slice_346 = None
    add_598: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_579, slice_347);  add_579 = slice_347 = None
    add_599: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_580, slice_348);  add_580 = slice_348 = None
    add_600: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_581, slice_349);  add_581 = slice_349 = None
    add_601: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_582, slice_350);  add_582 = slice_350 = None
    add_602: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_583, slice_351);  add_583 = slice_351 = None
    add_603: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_584, slice_352);  add_584 = slice_352 = None
    add_604: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_585, slice_353);  add_585 = slice_353 = None
    add_605: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_586, slice_354);  add_586 = slice_354 = None
    add_606: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_587, slice_355);  add_587 = slice_355 = None
    add_607: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_588, slice_356);  add_588 = slice_356 = None
    add_608: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_589, slice_357);  add_589 = slice_357 = None
    add_609: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_590, slice_358);  add_590 = slice_358 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(add_609, relu_68, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_609 = primals_207 = None
    getitem_155: "f32[4, 128, 14, 14]" = convolution_backward_51[0]
    getitem_156: "f32[32, 128, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    le_52: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    where_52: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, getitem_155);  le_52 = getitem_155 = None
    add_610: "f32[128]" = torch.ops.aten.add.Tensor(primals_570, 1e-05);  primals_570 = None
    rsqrt_52: "f32[128]" = torch.ops.aten.rsqrt.default(add_610);  add_610 = None
    unsqueeze_1592: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_569, 0);  primals_569 = None
    unsqueeze_1593: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 2);  unsqueeze_1592 = None
    unsqueeze_1594: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 3);  unsqueeze_1593 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_173: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_1594);  convolution_67 = unsqueeze_1594 = None
    mul_779: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_173);  sub_173 = None
    sum_107: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
    mul_784: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_205);  primals_205 = None
    unsqueeze_1601: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_1602: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
    unsqueeze_1603: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
    mul_785: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, unsqueeze_1603);  where_52 = unsqueeze_1603 = None
    mul_786: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, rsqrt_52);  sum_107 = rsqrt_52 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_785, relu_67, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_204 = None
    getitem_158: "f32[4, 704, 14, 14]" = convolution_backward_52[0]
    getitem_159: "f32[128, 704, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    le_53: "b8[4, 704, 14, 14]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
    where_53: "f32[4, 704, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, getitem_158);  le_53 = getitem_158 = None
    add_611: "f32[704]" = torch.ops.aten.add.Tensor(primals_567, 1e-05);  primals_567 = None
    rsqrt_53: "f32[704]" = torch.ops.aten.rsqrt.default(add_611);  add_611 = None
    unsqueeze_1604: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(primals_566, 0);  primals_566 = None
    unsqueeze_1605: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 2);  unsqueeze_1604 = None
    unsqueeze_1606: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 3);  unsqueeze_1605 = None
    sum_108: "f32[704]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_174: "f32[4, 704, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_1606);  cat_31 = unsqueeze_1606 = None
    mul_787: "f32[4, 704, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_174);  sub_174 = None
    sum_109: "f32[704]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_792: "f32[704]" = torch.ops.aten.mul.Tensor(rsqrt_53, primals_202);  primals_202 = None
    unsqueeze_1613: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_1614: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
    unsqueeze_1615: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
    mul_793: "f32[4, 704, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, unsqueeze_1615);  where_53 = unsqueeze_1615 = None
    mul_794: "f32[704]" = torch.ops.aten.mul.Tensor(sum_109, rsqrt_53);  sum_109 = rsqrt_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_359: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 0, 256)
    slice_360: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 256, 288)
    slice_361: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 288, 320)
    slice_362: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 320, 352)
    slice_363: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 352, 384)
    slice_364: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 384, 416)
    slice_365: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 416, 448)
    slice_366: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 448, 480)
    slice_367: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 480, 512)
    slice_368: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 512, 544)
    slice_369: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 544, 576)
    slice_370: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 576, 608)
    slice_371: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 608, 640)
    slice_372: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 640, 672)
    slice_373: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_793, 1, 672, 704);  mul_793 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_612: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_594, slice_359);  add_594 = slice_359 = None
    add_613: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_595, slice_360);  add_595 = slice_360 = None
    add_614: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_596, slice_361);  add_596 = slice_361 = None
    add_615: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_597, slice_362);  add_597 = slice_362 = None
    add_616: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_598, slice_363);  add_598 = slice_363 = None
    add_617: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_599, slice_364);  add_599 = slice_364 = None
    add_618: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_600, slice_365);  add_600 = slice_365 = None
    add_619: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_601, slice_366);  add_601 = slice_366 = None
    add_620: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_602, slice_367);  add_602 = slice_367 = None
    add_621: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_603, slice_368);  add_603 = slice_368 = None
    add_622: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_604, slice_369);  add_604 = slice_369 = None
    add_623: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_605, slice_370);  add_605 = slice_370 = None
    add_624: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_606, slice_371);  add_606 = slice_371 = None
    add_625: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_607, slice_372);  add_607 = slice_372 = None
    add_626: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_608, slice_373);  add_608 = slice_373 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(add_626, relu_66, primals_201, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_626 = primals_201 = None
    getitem_161: "f32[4, 128, 14, 14]" = convolution_backward_53[0]
    getitem_162: "f32[32, 128, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    le_54: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
    where_54: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, getitem_161);  le_54 = getitem_161 = None
    add_627: "f32[128]" = torch.ops.aten.add.Tensor(primals_564, 1e-05);  primals_564 = None
    rsqrt_54: "f32[128]" = torch.ops.aten.rsqrt.default(add_627);  add_627 = None
    unsqueeze_1616: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_563, 0);  primals_563 = None
    unsqueeze_1617: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 2);  unsqueeze_1616 = None
    unsqueeze_1618: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 3);  unsqueeze_1617 = None
    sum_110: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_175: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_1618);  convolution_65 = unsqueeze_1618 = None
    mul_795: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_175);  sub_175 = None
    sum_111: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
    mul_800: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_54, primals_199);  primals_199 = None
    unsqueeze_1625: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_1626: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
    unsqueeze_1627: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
    mul_801: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, unsqueeze_1627);  where_54 = unsqueeze_1627 = None
    mul_802: "f32[128]" = torch.ops.aten.mul.Tensor(sum_111, rsqrt_54);  sum_111 = rsqrt_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_801, relu_65, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_801 = primals_198 = None
    getitem_164: "f32[4, 672, 14, 14]" = convolution_backward_54[0]
    getitem_165: "f32[128, 672, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    le_55: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_55: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, getitem_164);  le_55 = getitem_164 = None
    add_628: "f32[672]" = torch.ops.aten.add.Tensor(primals_561, 1e-05);  primals_561 = None
    rsqrt_55: "f32[672]" = torch.ops.aten.rsqrt.default(add_628);  add_628 = None
    unsqueeze_1628: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_560, 0);  primals_560 = None
    unsqueeze_1629: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 2);  unsqueeze_1628 = None
    unsqueeze_1630: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 3);  unsqueeze_1629 = None
    sum_112: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_176: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_1630);  cat_30 = unsqueeze_1630 = None
    mul_803: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_176);  sub_176 = None
    sum_113: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3]);  mul_803 = None
    mul_808: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_55, primals_196);  primals_196 = None
    unsqueeze_1637: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_1638: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    mul_809: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, unsqueeze_1639);  where_55 = unsqueeze_1639 = None
    mul_810: "f32[672]" = torch.ops.aten.mul.Tensor(sum_113, rsqrt_55);  sum_113 = rsqrt_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_374: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 0, 256)
    slice_375: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 256, 288)
    slice_376: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 288, 320)
    slice_377: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 320, 352)
    slice_378: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 352, 384)
    slice_379: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 384, 416)
    slice_380: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 416, 448)
    slice_381: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 448, 480)
    slice_382: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 480, 512)
    slice_383: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 512, 544)
    slice_384: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 544, 576)
    slice_385: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 576, 608)
    slice_386: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 608, 640)
    slice_387: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_809, 1, 640, 672);  mul_809 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_629: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_612, slice_374);  add_612 = slice_374 = None
    add_630: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_613, slice_375);  add_613 = slice_375 = None
    add_631: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_614, slice_376);  add_614 = slice_376 = None
    add_632: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_615, slice_377);  add_615 = slice_377 = None
    add_633: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_616, slice_378);  add_616 = slice_378 = None
    add_634: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_617, slice_379);  add_617 = slice_379 = None
    add_635: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_618, slice_380);  add_618 = slice_380 = None
    add_636: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_619, slice_381);  add_619 = slice_381 = None
    add_637: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_620, slice_382);  add_620 = slice_382 = None
    add_638: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_621, slice_383);  add_621 = slice_383 = None
    add_639: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_622, slice_384);  add_622 = slice_384 = None
    add_640: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_623, slice_385);  add_623 = slice_385 = None
    add_641: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_624, slice_386);  add_624 = slice_386 = None
    add_642: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_625, slice_387);  add_625 = slice_387 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(add_642, relu_64, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_642 = primals_195 = None
    getitem_167: "f32[4, 128, 14, 14]" = convolution_backward_55[0]
    getitem_168: "f32[32, 128, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    le_56: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
    where_56: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, getitem_167);  le_56 = getitem_167 = None
    add_643: "f32[128]" = torch.ops.aten.add.Tensor(primals_558, 1e-05);  primals_558 = None
    rsqrt_56: "f32[128]" = torch.ops.aten.rsqrt.default(add_643);  add_643 = None
    unsqueeze_1640: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_557, 0);  primals_557 = None
    unsqueeze_1641: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
    unsqueeze_1642: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None
    sum_114: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_177: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_1642);  convolution_63 = unsqueeze_1642 = None
    mul_811: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_177);  sub_177 = None
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
    mul_816: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_56, primals_193);  primals_193 = None
    unsqueeze_1649: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_1650: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
    unsqueeze_1651: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
    mul_817: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, unsqueeze_1651);  where_56 = unsqueeze_1651 = None
    mul_818: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, rsqrt_56);  sum_115 = rsqrt_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_817, relu_63, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_817 = primals_192 = None
    getitem_170: "f32[4, 640, 14, 14]" = convolution_backward_56[0]
    getitem_171: "f32[128, 640, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    le_57: "b8[4, 640, 14, 14]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    where_57: "f32[4, 640, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, getitem_170);  le_57 = getitem_170 = None
    add_644: "f32[640]" = torch.ops.aten.add.Tensor(primals_555, 1e-05);  primals_555 = None
    rsqrt_57: "f32[640]" = torch.ops.aten.rsqrt.default(add_644);  add_644 = None
    unsqueeze_1652: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(primals_554, 0);  primals_554 = None
    unsqueeze_1653: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 2);  unsqueeze_1652 = None
    unsqueeze_1654: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 3);  unsqueeze_1653 = None
    sum_116: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_178: "f32[4, 640, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_1654);  cat_29 = unsqueeze_1654 = None
    mul_819: "f32[4, 640, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_178);  sub_178 = None
    sum_117: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_819, [0, 2, 3]);  mul_819 = None
    mul_824: "f32[640]" = torch.ops.aten.mul.Tensor(rsqrt_57, primals_190);  primals_190 = None
    unsqueeze_1661: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_1662: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
    unsqueeze_1663: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
    mul_825: "f32[4, 640, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, unsqueeze_1663);  where_57 = unsqueeze_1663 = None
    mul_826: "f32[640]" = torch.ops.aten.mul.Tensor(sum_117, rsqrt_57);  sum_117 = rsqrt_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_388: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 0, 256)
    slice_389: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 256, 288)
    slice_390: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 288, 320)
    slice_391: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 320, 352)
    slice_392: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 352, 384)
    slice_393: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 384, 416)
    slice_394: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 416, 448)
    slice_395: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 448, 480)
    slice_396: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 480, 512)
    slice_397: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 512, 544)
    slice_398: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 544, 576)
    slice_399: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 576, 608)
    slice_400: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 608, 640);  mul_825 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_645: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_629, slice_388);  add_629 = slice_388 = None
    add_646: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_630, slice_389);  add_630 = slice_389 = None
    add_647: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_631, slice_390);  add_631 = slice_390 = None
    add_648: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_632, slice_391);  add_632 = slice_391 = None
    add_649: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_633, slice_392);  add_633 = slice_392 = None
    add_650: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_634, slice_393);  add_634 = slice_393 = None
    add_651: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_635, slice_394);  add_635 = slice_394 = None
    add_652: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_636, slice_395);  add_636 = slice_395 = None
    add_653: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_637, slice_396);  add_637 = slice_396 = None
    add_654: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_638, slice_397);  add_638 = slice_397 = None
    add_655: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_639, slice_398);  add_639 = slice_398 = None
    add_656: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_640, slice_399);  add_640 = slice_399 = None
    add_657: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_641, slice_400);  add_641 = slice_400 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(add_657, relu_62, primals_189, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_657 = primals_189 = None
    getitem_173: "f32[4, 128, 14, 14]" = convolution_backward_57[0]
    getitem_174: "f32[32, 128, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    le_58: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_58: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, getitem_173);  le_58 = getitem_173 = None
    add_658: "f32[128]" = torch.ops.aten.add.Tensor(primals_552, 1e-05);  primals_552 = None
    rsqrt_58: "f32[128]" = torch.ops.aten.rsqrt.default(add_658);  add_658 = None
    unsqueeze_1664: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_551, 0);  primals_551 = None
    unsqueeze_1665: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, 2);  unsqueeze_1664 = None
    unsqueeze_1666: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 3);  unsqueeze_1665 = None
    sum_118: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_179: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_1666);  convolution_61 = unsqueeze_1666 = None
    mul_827: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_179);  sub_179 = None
    sum_119: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
    mul_832: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_58, primals_187);  primals_187 = None
    unsqueeze_1673: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_1674: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 2);  unsqueeze_1673 = None
    unsqueeze_1675: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 3);  unsqueeze_1674 = None
    mul_833: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, unsqueeze_1675);  where_58 = unsqueeze_1675 = None
    mul_834: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, rsqrt_58);  sum_119 = rsqrt_58 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_833, relu_61, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_833 = primals_186 = None
    getitem_176: "f32[4, 608, 14, 14]" = convolution_backward_58[0]
    getitem_177: "f32[128, 608, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    le_59: "b8[4, 608, 14, 14]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
    where_59: "f32[4, 608, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, getitem_176);  le_59 = getitem_176 = None
    add_659: "f32[608]" = torch.ops.aten.add.Tensor(primals_549, 1e-05);  primals_549 = None
    rsqrt_59: "f32[608]" = torch.ops.aten.rsqrt.default(add_659);  add_659 = None
    unsqueeze_1676: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(primals_548, 0);  primals_548 = None
    unsqueeze_1677: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, 2);  unsqueeze_1676 = None
    unsqueeze_1678: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 3);  unsqueeze_1677 = None
    sum_120: "f32[608]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_180: "f32[4, 608, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_1678);  cat_28 = unsqueeze_1678 = None
    mul_835: "f32[4, 608, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_180);  sub_180 = None
    sum_121: "f32[608]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2, 3]);  mul_835 = None
    mul_840: "f32[608]" = torch.ops.aten.mul.Tensor(rsqrt_59, primals_184);  primals_184 = None
    unsqueeze_1685: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_1686: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 2);  unsqueeze_1685 = None
    unsqueeze_1687: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 3);  unsqueeze_1686 = None
    mul_841: "f32[4, 608, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, unsqueeze_1687);  where_59 = unsqueeze_1687 = None
    mul_842: "f32[608]" = torch.ops.aten.mul.Tensor(sum_121, rsqrt_59);  sum_121 = rsqrt_59 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_401: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 0, 256)
    slice_402: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 256, 288)
    slice_403: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 288, 320)
    slice_404: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 320, 352)
    slice_405: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 352, 384)
    slice_406: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 384, 416)
    slice_407: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 416, 448)
    slice_408: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 448, 480)
    slice_409: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 480, 512)
    slice_410: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 512, 544)
    slice_411: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 544, 576)
    slice_412: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_841, 1, 576, 608);  mul_841 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_660: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_645, slice_401);  add_645 = slice_401 = None
    add_661: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_646, slice_402);  add_646 = slice_402 = None
    add_662: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_647, slice_403);  add_647 = slice_403 = None
    add_663: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_648, slice_404);  add_648 = slice_404 = None
    add_664: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_649, slice_405);  add_649 = slice_405 = None
    add_665: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_650, slice_406);  add_650 = slice_406 = None
    add_666: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_651, slice_407);  add_651 = slice_407 = None
    add_667: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_652, slice_408);  add_652 = slice_408 = None
    add_668: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_653, slice_409);  add_653 = slice_409 = None
    add_669: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_654, slice_410);  add_654 = slice_410 = None
    add_670: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_655, slice_411);  add_655 = slice_411 = None
    add_671: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_656, slice_412);  add_656 = slice_412 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_671, relu_60, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_671 = primals_183 = None
    getitem_179: "f32[4, 128, 14, 14]" = convolution_backward_59[0]
    getitem_180: "f32[32, 128, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    le_60: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_60: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, getitem_179);  le_60 = getitem_179 = None
    add_672: "f32[128]" = torch.ops.aten.add.Tensor(primals_546, 1e-05);  primals_546 = None
    rsqrt_60: "f32[128]" = torch.ops.aten.rsqrt.default(add_672);  add_672 = None
    unsqueeze_1688: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_545, 0);  primals_545 = None
    unsqueeze_1689: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 2);  unsqueeze_1688 = None
    unsqueeze_1690: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1689, 3);  unsqueeze_1689 = None
    sum_122: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_181: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_1690);  convolution_59 = unsqueeze_1690 = None
    mul_843: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_181);  sub_181 = None
    sum_123: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_843, [0, 2, 3]);  mul_843 = None
    mul_848: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_60, primals_181);  primals_181 = None
    unsqueeze_1697: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_1698: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 2);  unsqueeze_1697 = None
    unsqueeze_1699: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 3);  unsqueeze_1698 = None
    mul_849: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, unsqueeze_1699);  where_60 = unsqueeze_1699 = None
    mul_850: "f32[128]" = torch.ops.aten.mul.Tensor(sum_123, rsqrt_60);  sum_123 = rsqrt_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_849, relu_59, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_849 = primals_180 = None
    getitem_182: "f32[4, 576, 14, 14]" = convolution_backward_60[0]
    getitem_183: "f32[128, 576, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    le_61: "b8[4, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    where_61: "f32[4, 576, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, getitem_182);  le_61 = getitem_182 = None
    add_673: "f32[576]" = torch.ops.aten.add.Tensor(primals_543, 1e-05);  primals_543 = None
    rsqrt_61: "f32[576]" = torch.ops.aten.rsqrt.default(add_673);  add_673 = None
    unsqueeze_1700: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(primals_542, 0);  primals_542 = None
    unsqueeze_1701: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 2);  unsqueeze_1700 = None
    unsqueeze_1702: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 3);  unsqueeze_1701 = None
    sum_124: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_182: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_1702);  cat_27 = unsqueeze_1702 = None
    mul_851: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_182);  sub_182 = None
    sum_125: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
    mul_856: "f32[576]" = torch.ops.aten.mul.Tensor(rsqrt_61, primals_178);  primals_178 = None
    unsqueeze_1709: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_1710: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 2);  unsqueeze_1709 = None
    unsqueeze_1711: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 3);  unsqueeze_1710 = None
    mul_857: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, unsqueeze_1711);  where_61 = unsqueeze_1711 = None
    mul_858: "f32[576]" = torch.ops.aten.mul.Tensor(sum_125, rsqrt_61);  sum_125 = rsqrt_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_413: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 0, 256)
    slice_414: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 256, 288)
    slice_415: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 288, 320)
    slice_416: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 320, 352)
    slice_417: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 352, 384)
    slice_418: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 384, 416)
    slice_419: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 416, 448)
    slice_420: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 448, 480)
    slice_421: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 480, 512)
    slice_422: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 512, 544)
    slice_423: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_857, 1, 544, 576);  mul_857 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_674: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_660, slice_413);  add_660 = slice_413 = None
    add_675: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_661, slice_414);  add_661 = slice_414 = None
    add_676: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_662, slice_415);  add_662 = slice_415 = None
    add_677: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_663, slice_416);  add_663 = slice_416 = None
    add_678: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_664, slice_417);  add_664 = slice_417 = None
    add_679: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_665, slice_418);  add_665 = slice_418 = None
    add_680: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_666, slice_419);  add_666 = slice_419 = None
    add_681: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_667, slice_420);  add_667 = slice_420 = None
    add_682: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_668, slice_421);  add_668 = slice_421 = None
    add_683: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_669, slice_422);  add_669 = slice_422 = None
    add_684: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_670, slice_423);  add_670 = slice_423 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(add_684, relu_58, primals_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_684 = primals_177 = None
    getitem_185: "f32[4, 128, 14, 14]" = convolution_backward_61[0]
    getitem_186: "f32[32, 128, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    le_62: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    where_62: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_62, full_default, getitem_185);  le_62 = getitem_185 = None
    add_685: "f32[128]" = torch.ops.aten.add.Tensor(primals_540, 1e-05);  primals_540 = None
    rsqrt_62: "f32[128]" = torch.ops.aten.rsqrt.default(add_685);  add_685 = None
    unsqueeze_1712: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_539, 0);  primals_539 = None
    unsqueeze_1713: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 2);  unsqueeze_1712 = None
    unsqueeze_1714: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 3);  unsqueeze_1713 = None
    sum_126: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_183: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_1714);  convolution_57 = unsqueeze_1714 = None
    mul_859: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_183);  sub_183 = None
    sum_127: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_859, [0, 2, 3]);  mul_859 = None
    mul_864: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_62, primals_175);  primals_175 = None
    unsqueeze_1721: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_1722: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 2);  unsqueeze_1721 = None
    unsqueeze_1723: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 3);  unsqueeze_1722 = None
    mul_865: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, unsqueeze_1723);  where_62 = unsqueeze_1723 = None
    mul_866: "f32[128]" = torch.ops.aten.mul.Tensor(sum_127, rsqrt_62);  sum_127 = rsqrt_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_865, relu_57, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_865 = primals_174 = None
    getitem_188: "f32[4, 544, 14, 14]" = convolution_backward_62[0]
    getitem_189: "f32[128, 544, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    le_63: "b8[4, 544, 14, 14]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
    where_63: "f32[4, 544, 14, 14]" = torch.ops.aten.where.self(le_63, full_default, getitem_188);  le_63 = getitem_188 = None
    add_686: "f32[544]" = torch.ops.aten.add.Tensor(primals_537, 1e-05);  primals_537 = None
    rsqrt_63: "f32[544]" = torch.ops.aten.rsqrt.default(add_686);  add_686 = None
    unsqueeze_1724: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(primals_536, 0);  primals_536 = None
    unsqueeze_1725: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 2);  unsqueeze_1724 = None
    unsqueeze_1726: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 3);  unsqueeze_1725 = None
    sum_128: "f32[544]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_184: "f32[4, 544, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_1726);  cat_26 = unsqueeze_1726 = None
    mul_867: "f32[4, 544, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_184);  sub_184 = None
    sum_129: "f32[544]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
    mul_872: "f32[544]" = torch.ops.aten.mul.Tensor(rsqrt_63, primals_172);  primals_172 = None
    unsqueeze_1733: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_1734: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 2);  unsqueeze_1733 = None
    unsqueeze_1735: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 3);  unsqueeze_1734 = None
    mul_873: "f32[4, 544, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, unsqueeze_1735);  where_63 = unsqueeze_1735 = None
    mul_874: "f32[544]" = torch.ops.aten.mul.Tensor(sum_129, rsqrt_63);  sum_129 = rsqrt_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_424: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 0, 256)
    slice_425: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 256, 288)
    slice_426: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 288, 320)
    slice_427: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 320, 352)
    slice_428: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 352, 384)
    slice_429: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 384, 416)
    slice_430: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 416, 448)
    slice_431: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 448, 480)
    slice_432: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 480, 512)
    slice_433: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_873, 1, 512, 544);  mul_873 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_687: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_674, slice_424);  add_674 = slice_424 = None
    add_688: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_675, slice_425);  add_675 = slice_425 = None
    add_689: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_676, slice_426);  add_676 = slice_426 = None
    add_690: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_677, slice_427);  add_677 = slice_427 = None
    add_691: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_678, slice_428);  add_678 = slice_428 = None
    add_692: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_679, slice_429);  add_679 = slice_429 = None
    add_693: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_680, slice_430);  add_680 = slice_430 = None
    add_694: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_681, slice_431);  add_681 = slice_431 = None
    add_695: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_682, slice_432);  add_682 = slice_432 = None
    add_696: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_683, slice_433);  add_683 = slice_433 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(add_696, relu_56, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_696 = primals_171 = None
    getitem_191: "f32[4, 128, 14, 14]" = convolution_backward_63[0]
    getitem_192: "f32[32, 128, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    le_64: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_64: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_64, full_default, getitem_191);  le_64 = getitem_191 = None
    add_697: "f32[128]" = torch.ops.aten.add.Tensor(primals_534, 1e-05);  primals_534 = None
    rsqrt_64: "f32[128]" = torch.ops.aten.rsqrt.default(add_697);  add_697 = None
    unsqueeze_1736: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_533, 0);  primals_533 = None
    unsqueeze_1737: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, 2);  unsqueeze_1736 = None
    unsqueeze_1738: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 3);  unsqueeze_1737 = None
    sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_185: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1738);  convolution_55 = unsqueeze_1738 = None
    mul_875: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_185);  sub_185 = None
    sum_131: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_875, [0, 2, 3]);  mul_875 = None
    mul_880: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_64, primals_169);  primals_169 = None
    unsqueeze_1745: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_1746: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 2);  unsqueeze_1745 = None
    unsqueeze_1747: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 3);  unsqueeze_1746 = None
    mul_881: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_1747);  where_64 = unsqueeze_1747 = None
    mul_882: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, rsqrt_64);  sum_131 = rsqrt_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_881, relu_55, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_881 = primals_168 = None
    getitem_194: "f32[4, 512, 14, 14]" = convolution_backward_64[0]
    getitem_195: "f32[128, 512, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    le_65: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_65: "f32[4, 512, 14, 14]" = torch.ops.aten.where.self(le_65, full_default, getitem_194);  le_65 = getitem_194 = None
    add_698: "f32[512]" = torch.ops.aten.add.Tensor(primals_531, 1e-05);  primals_531 = None
    rsqrt_65: "f32[512]" = torch.ops.aten.rsqrt.default(add_698);  add_698 = None
    unsqueeze_1748: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_530, 0);  primals_530 = None
    unsqueeze_1749: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 2);  unsqueeze_1748 = None
    unsqueeze_1750: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1749, 3);  unsqueeze_1749 = None
    sum_132: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_186: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_1750);  cat_25 = unsqueeze_1750 = None
    mul_883: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_186);  sub_186 = None
    sum_133: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
    mul_888: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_65, primals_166);  primals_166 = None
    unsqueeze_1757: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_1758: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 2);  unsqueeze_1757 = None
    unsqueeze_1759: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 3);  unsqueeze_1758 = None
    mul_889: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, unsqueeze_1759);  where_65 = unsqueeze_1759 = None
    mul_890: "f32[512]" = torch.ops.aten.mul.Tensor(sum_133, rsqrt_65);  sum_133 = rsqrt_65 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_434: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 0, 256)
    slice_435: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 256, 288)
    slice_436: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 288, 320)
    slice_437: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 320, 352)
    slice_438: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 352, 384)
    slice_439: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 384, 416)
    slice_440: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 416, 448)
    slice_441: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 448, 480)
    slice_442: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_889, 1, 480, 512);  mul_889 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_699: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_687, slice_434);  add_687 = slice_434 = None
    add_700: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_688, slice_435);  add_688 = slice_435 = None
    add_701: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_689, slice_436);  add_689 = slice_436 = None
    add_702: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_690, slice_437);  add_690 = slice_437 = None
    add_703: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_691, slice_438);  add_691 = slice_438 = None
    add_704: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_692, slice_439);  add_692 = slice_439 = None
    add_705: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_693, slice_440);  add_693 = slice_440 = None
    add_706: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_694, slice_441);  add_694 = slice_441 = None
    add_707: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_695, slice_442);  add_695 = slice_442 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_707, relu_54, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_707 = primals_165 = None
    getitem_197: "f32[4, 128, 14, 14]" = convolution_backward_65[0]
    getitem_198: "f32[32, 128, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    le_66: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_66: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_66, full_default, getitem_197);  le_66 = getitem_197 = None
    add_708: "f32[128]" = torch.ops.aten.add.Tensor(primals_528, 1e-05);  primals_528 = None
    rsqrt_66: "f32[128]" = torch.ops.aten.rsqrt.default(add_708);  add_708 = None
    unsqueeze_1760: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_527, 0);  primals_527 = None
    unsqueeze_1761: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 2);  unsqueeze_1760 = None
    unsqueeze_1762: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1761, 3);  unsqueeze_1761 = None
    sum_134: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_187: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1762);  convolution_53 = unsqueeze_1762 = None
    mul_891: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_187);  sub_187 = None
    sum_135: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_891, [0, 2, 3]);  mul_891 = None
    mul_896: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_66, primals_163);  primals_163 = None
    unsqueeze_1769: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_1770: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 2);  unsqueeze_1769 = None
    unsqueeze_1771: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 3);  unsqueeze_1770 = None
    mul_897: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, unsqueeze_1771);  where_66 = unsqueeze_1771 = None
    mul_898: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, rsqrt_66);  sum_135 = rsqrt_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_897, relu_53, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_897 = primals_162 = None
    getitem_200: "f32[4, 480, 14, 14]" = convolution_backward_66[0]
    getitem_201: "f32[128, 480, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    le_67: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_67: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_67, full_default, getitem_200);  le_67 = getitem_200 = None
    add_709: "f32[480]" = torch.ops.aten.add.Tensor(primals_525, 1e-05);  primals_525 = None
    rsqrt_67: "f32[480]" = torch.ops.aten.rsqrt.default(add_709);  add_709 = None
    unsqueeze_1772: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_524, 0);  primals_524 = None
    unsqueeze_1773: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 2);  unsqueeze_1772 = None
    unsqueeze_1774: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1773, 3);  unsqueeze_1773 = None
    sum_136: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_188: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_1774);  cat_24 = unsqueeze_1774 = None
    mul_899: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_188);  sub_188 = None
    sum_137: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 2, 3]);  mul_899 = None
    mul_904: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_67, primals_160);  primals_160 = None
    unsqueeze_1781: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_1782: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 2);  unsqueeze_1781 = None
    unsqueeze_1783: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 3);  unsqueeze_1782 = None
    mul_905: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, unsqueeze_1783);  where_67 = unsqueeze_1783 = None
    mul_906: "f32[480]" = torch.ops.aten.mul.Tensor(sum_137, rsqrt_67);  sum_137 = rsqrt_67 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_443: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 0, 256)
    slice_444: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 256, 288)
    slice_445: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 288, 320)
    slice_446: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 320, 352)
    slice_447: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 352, 384)
    slice_448: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 384, 416)
    slice_449: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 416, 448)
    slice_450: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 448, 480);  mul_905 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_710: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_699, slice_443);  add_699 = slice_443 = None
    add_711: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_700, slice_444);  add_700 = slice_444 = None
    add_712: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_701, slice_445);  add_701 = slice_445 = None
    add_713: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_702, slice_446);  add_702 = slice_446 = None
    add_714: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_703, slice_447);  add_703 = slice_447 = None
    add_715: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_704, slice_448);  add_704 = slice_448 = None
    add_716: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_705, slice_449);  add_705 = slice_449 = None
    add_717: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_706, slice_450);  add_706 = slice_450 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(add_717, relu_52, primals_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_717 = primals_159 = None
    getitem_203: "f32[4, 128, 14, 14]" = convolution_backward_67[0]
    getitem_204: "f32[32, 128, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    le_68: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_68: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_68, full_default, getitem_203);  le_68 = getitem_203 = None
    add_718: "f32[128]" = torch.ops.aten.add.Tensor(primals_522, 1e-05);  primals_522 = None
    rsqrt_68: "f32[128]" = torch.ops.aten.rsqrt.default(add_718);  add_718 = None
    unsqueeze_1784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_521, 0);  primals_521 = None
    unsqueeze_1785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 2);  unsqueeze_1784 = None
    unsqueeze_1786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1785, 3);  unsqueeze_1785 = None
    sum_138: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_189: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1786);  convolution_51 = unsqueeze_1786 = None
    mul_907: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_189);  sub_189 = None
    sum_139: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_907, [0, 2, 3]);  mul_907 = None
    mul_912: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_68, primals_157);  primals_157 = None
    unsqueeze_1793: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_1794: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 2);  unsqueeze_1793 = None
    unsqueeze_1795: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 3);  unsqueeze_1794 = None
    mul_913: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, unsqueeze_1795);  where_68 = unsqueeze_1795 = None
    mul_914: "f32[128]" = torch.ops.aten.mul.Tensor(sum_139, rsqrt_68);  sum_139 = rsqrt_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_913, relu_51, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_913 = primals_156 = None
    getitem_206: "f32[4, 448, 14, 14]" = convolution_backward_68[0]
    getitem_207: "f32[128, 448, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    le_69: "b8[4, 448, 14, 14]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_69: "f32[4, 448, 14, 14]" = torch.ops.aten.where.self(le_69, full_default, getitem_206);  le_69 = getitem_206 = None
    add_719: "f32[448]" = torch.ops.aten.add.Tensor(primals_519, 1e-05);  primals_519 = None
    rsqrt_69: "f32[448]" = torch.ops.aten.rsqrt.default(add_719);  add_719 = None
    unsqueeze_1796: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_518, 0);  primals_518 = None
    unsqueeze_1797: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, 2);  unsqueeze_1796 = None
    unsqueeze_1798: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1797, 3);  unsqueeze_1797 = None
    sum_140: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_190: "f32[4, 448, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_1798);  cat_23 = unsqueeze_1798 = None
    mul_915: "f32[4, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, sub_190);  sub_190 = None
    sum_141: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3]);  mul_915 = None
    mul_920: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_69, primals_154);  primals_154 = None
    unsqueeze_1805: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_1806: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 2);  unsqueeze_1805 = None
    unsqueeze_1807: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 3);  unsqueeze_1806 = None
    mul_921: "f32[4, 448, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, unsqueeze_1807);  where_69 = unsqueeze_1807 = None
    mul_922: "f32[448]" = torch.ops.aten.mul.Tensor(sum_141, rsqrt_69);  sum_141 = rsqrt_69 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_451: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 0, 256)
    slice_452: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 256, 288)
    slice_453: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 288, 320)
    slice_454: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 320, 352)
    slice_455: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 352, 384)
    slice_456: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 384, 416)
    slice_457: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_921, 1, 416, 448);  mul_921 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_720: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_710, slice_451);  add_710 = slice_451 = None
    add_721: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_711, slice_452);  add_711 = slice_452 = None
    add_722: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_712, slice_453);  add_712 = slice_453 = None
    add_723: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_713, slice_454);  add_713 = slice_454 = None
    add_724: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_714, slice_455);  add_714 = slice_455 = None
    add_725: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_715, slice_456);  add_715 = slice_456 = None
    add_726: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_716, slice_457);  add_716 = slice_457 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_726, relu_50, primals_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_726 = primals_153 = None
    getitem_209: "f32[4, 128, 14, 14]" = convolution_backward_69[0]
    getitem_210: "f32[32, 128, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    le_70: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_70: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_70, full_default, getitem_209);  le_70 = getitem_209 = None
    add_727: "f32[128]" = torch.ops.aten.add.Tensor(primals_516, 1e-05);  primals_516 = None
    rsqrt_70: "f32[128]" = torch.ops.aten.rsqrt.default(add_727);  add_727 = None
    unsqueeze_1808: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_515, 0);  primals_515 = None
    unsqueeze_1809: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, 2);  unsqueeze_1808 = None
    unsqueeze_1810: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1809, 3);  unsqueeze_1809 = None
    sum_142: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_191: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1810);  convolution_49 = unsqueeze_1810 = None
    mul_923: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, sub_191);  sub_191 = None
    sum_143: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 2, 3]);  mul_923 = None
    mul_928: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_70, primals_151);  primals_151 = None
    unsqueeze_1817: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_1818: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 2);  unsqueeze_1817 = None
    unsqueeze_1819: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 3);  unsqueeze_1818 = None
    mul_929: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, unsqueeze_1819);  where_70 = unsqueeze_1819 = None
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(sum_143, rsqrt_70);  sum_143 = rsqrt_70 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_929, relu_49, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_929 = primals_150 = None
    getitem_212: "f32[4, 416, 14, 14]" = convolution_backward_70[0]
    getitem_213: "f32[128, 416, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    le_71: "b8[4, 416, 14, 14]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_71: "f32[4, 416, 14, 14]" = torch.ops.aten.where.self(le_71, full_default, getitem_212);  le_71 = getitem_212 = None
    add_728: "f32[416]" = torch.ops.aten.add.Tensor(primals_513, 1e-05);  primals_513 = None
    rsqrt_71: "f32[416]" = torch.ops.aten.rsqrt.default(add_728);  add_728 = None
    unsqueeze_1820: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(primals_512, 0);  primals_512 = None
    unsqueeze_1821: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, 2);  unsqueeze_1820 = None
    unsqueeze_1822: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1821, 3);  unsqueeze_1821 = None
    sum_144: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_192: "f32[4, 416, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_1822);  cat_22 = unsqueeze_1822 = None
    mul_931: "f32[4, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_192);  sub_192 = None
    sum_145: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
    mul_936: "f32[416]" = torch.ops.aten.mul.Tensor(rsqrt_71, primals_148);  primals_148 = None
    unsqueeze_1829: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_1830: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 2);  unsqueeze_1829 = None
    unsqueeze_1831: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 3);  unsqueeze_1830 = None
    mul_937: "f32[4, 416, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, unsqueeze_1831);  where_71 = unsqueeze_1831 = None
    mul_938: "f32[416]" = torch.ops.aten.mul.Tensor(sum_145, rsqrt_71);  sum_145 = rsqrt_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_458: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 0, 256)
    slice_459: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 256, 288)
    slice_460: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 288, 320)
    slice_461: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 320, 352)
    slice_462: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 352, 384)
    slice_463: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_937, 1, 384, 416);  mul_937 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_729: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_720, slice_458);  add_720 = slice_458 = None
    add_730: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_721, slice_459);  add_721 = slice_459 = None
    add_731: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_722, slice_460);  add_722 = slice_460 = None
    add_732: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_723, slice_461);  add_723 = slice_461 = None
    add_733: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_724, slice_462);  add_724 = slice_462 = None
    add_734: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_725, slice_463);  add_725 = slice_463 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(add_734, relu_48, primals_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_734 = primals_147 = None
    getitem_215: "f32[4, 128, 14, 14]" = convolution_backward_71[0]
    getitem_216: "f32[32, 128, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    le_72: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_72: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_72, full_default, getitem_215);  le_72 = getitem_215 = None
    add_735: "f32[128]" = torch.ops.aten.add.Tensor(primals_510, 1e-05);  primals_510 = None
    rsqrt_72: "f32[128]" = torch.ops.aten.rsqrt.default(add_735);  add_735 = None
    unsqueeze_1832: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_509, 0);  primals_509 = None
    unsqueeze_1833: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, 2);  unsqueeze_1832 = None
    unsqueeze_1834: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1833, 3);  unsqueeze_1833 = None
    sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_193: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1834);  convolution_47 = unsqueeze_1834 = None
    mul_939: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_193);  sub_193 = None
    sum_147: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2, 3]);  mul_939 = None
    mul_944: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_72, primals_145);  primals_145 = None
    unsqueeze_1841: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_1842: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 2);  unsqueeze_1841 = None
    unsqueeze_1843: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, 3);  unsqueeze_1842 = None
    mul_945: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, unsqueeze_1843);  where_72 = unsqueeze_1843 = None
    mul_946: "f32[128]" = torch.ops.aten.mul.Tensor(sum_147, rsqrt_72);  sum_147 = rsqrt_72 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_945, relu_47, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_945 = primals_144 = None
    getitem_218: "f32[4, 384, 14, 14]" = convolution_backward_72[0]
    getitem_219: "f32[128, 384, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    le_73: "b8[4, 384, 14, 14]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_73: "f32[4, 384, 14, 14]" = torch.ops.aten.where.self(le_73, full_default, getitem_218);  le_73 = getitem_218 = None
    add_736: "f32[384]" = torch.ops.aten.add.Tensor(primals_507, 1e-05);  primals_507 = None
    rsqrt_73: "f32[384]" = torch.ops.aten.rsqrt.default(add_736);  add_736 = None
    unsqueeze_1844: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_506, 0);  primals_506 = None
    unsqueeze_1845: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 2);  unsqueeze_1844 = None
    unsqueeze_1846: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1845, 3);  unsqueeze_1845 = None
    sum_148: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_194: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_1846);  cat_21 = unsqueeze_1846 = None
    mul_947: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_194);  sub_194 = None
    sum_149: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3]);  mul_947 = None
    mul_952: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_73, primals_142);  primals_142 = None
    unsqueeze_1853: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_1854: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 2);  unsqueeze_1853 = None
    unsqueeze_1855: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, 3);  unsqueeze_1854 = None
    mul_953: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, unsqueeze_1855);  where_73 = unsqueeze_1855 = None
    mul_954: "f32[384]" = torch.ops.aten.mul.Tensor(sum_149, rsqrt_73);  sum_149 = rsqrt_73 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_464: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_953, 1, 0, 256)
    slice_465: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_953, 1, 256, 288)
    slice_466: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_953, 1, 288, 320)
    slice_467: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_953, 1, 320, 352)
    slice_468: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_953, 1, 352, 384);  mul_953 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_737: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_729, slice_464);  add_729 = slice_464 = None
    add_738: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_730, slice_465);  add_730 = slice_465 = None
    add_739: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_731, slice_466);  add_731 = slice_466 = None
    add_740: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_732, slice_467);  add_732 = slice_467 = None
    add_741: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_733, slice_468);  add_733 = slice_468 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(add_741, relu_46, primals_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_741 = primals_141 = None
    getitem_221: "f32[4, 128, 14, 14]" = convolution_backward_73[0]
    getitem_222: "f32[32, 128, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    le_74: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_74: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_74, full_default, getitem_221);  le_74 = getitem_221 = None
    add_742: "f32[128]" = torch.ops.aten.add.Tensor(primals_504, 1e-05);  primals_504 = None
    rsqrt_74: "f32[128]" = torch.ops.aten.rsqrt.default(add_742);  add_742 = None
    unsqueeze_1856: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_503, 0);  primals_503 = None
    unsqueeze_1857: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 2);  unsqueeze_1856 = None
    unsqueeze_1858: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1857, 3);  unsqueeze_1857 = None
    sum_150: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_195: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1858);  convolution_45 = unsqueeze_1858 = None
    mul_955: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, sub_195);  sub_195 = None
    sum_151: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
    mul_960: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_74, primals_139);  primals_139 = None
    unsqueeze_1865: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_1866: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 2);  unsqueeze_1865 = None
    unsqueeze_1867: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 3);  unsqueeze_1866 = None
    mul_961: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, unsqueeze_1867);  where_74 = unsqueeze_1867 = None
    mul_962: "f32[128]" = torch.ops.aten.mul.Tensor(sum_151, rsqrt_74);  sum_151 = rsqrt_74 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_961, relu_45, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_961 = primals_138 = None
    getitem_224: "f32[4, 352, 14, 14]" = convolution_backward_74[0]
    getitem_225: "f32[128, 352, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    le_75: "b8[4, 352, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_75: "f32[4, 352, 14, 14]" = torch.ops.aten.where.self(le_75, full_default, getitem_224);  le_75 = getitem_224 = None
    add_743: "f32[352]" = torch.ops.aten.add.Tensor(primals_501, 1e-05);  primals_501 = None
    rsqrt_75: "f32[352]" = torch.ops.aten.rsqrt.default(add_743);  add_743 = None
    unsqueeze_1868: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(primals_500, 0);  primals_500 = None
    unsqueeze_1869: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 2);  unsqueeze_1868 = None
    unsqueeze_1870: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1869, 3);  unsqueeze_1869 = None
    sum_152: "f32[352]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_196: "f32[4, 352, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_1870);  cat_20 = unsqueeze_1870 = None
    mul_963: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_196);  sub_196 = None
    sum_153: "f32[352]" = torch.ops.aten.sum.dim_IntList(mul_963, [0, 2, 3]);  mul_963 = None
    mul_968: "f32[352]" = torch.ops.aten.mul.Tensor(rsqrt_75, primals_136);  primals_136 = None
    unsqueeze_1877: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
    unsqueeze_1878: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 2);  unsqueeze_1877 = None
    unsqueeze_1879: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 3);  unsqueeze_1878 = None
    mul_969: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, unsqueeze_1879);  where_75 = unsqueeze_1879 = None
    mul_970: "f32[352]" = torch.ops.aten.mul.Tensor(sum_153, rsqrt_75);  sum_153 = rsqrt_75 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_469: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_969, 1, 0, 256)
    slice_470: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_969, 1, 256, 288)
    slice_471: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_969, 1, 288, 320)
    slice_472: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_969, 1, 320, 352);  mul_969 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_744: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_737, slice_469);  add_737 = slice_469 = None
    add_745: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_738, slice_470);  add_738 = slice_470 = None
    add_746: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_739, slice_471);  add_739 = slice_471 = None
    add_747: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_740, slice_472);  add_740 = slice_472 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(add_747, relu_44, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_747 = primals_135 = None
    getitem_227: "f32[4, 128, 14, 14]" = convolution_backward_75[0]
    getitem_228: "f32[32, 128, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    le_76: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_76: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_76, full_default, getitem_227);  le_76 = getitem_227 = None
    add_748: "f32[128]" = torch.ops.aten.add.Tensor(primals_498, 1e-05);  primals_498 = None
    rsqrt_76: "f32[128]" = torch.ops.aten.rsqrt.default(add_748);  add_748 = None
    unsqueeze_1880: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_497, 0);  primals_497 = None
    unsqueeze_1881: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 2);  unsqueeze_1880 = None
    unsqueeze_1882: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1881, 3);  unsqueeze_1881 = None
    sum_154: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_197: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1882);  convolution_43 = unsqueeze_1882 = None
    mul_971: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, sub_197);  sub_197 = None
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
    mul_976: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_76, primals_133);  primals_133 = None
    unsqueeze_1889: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_1890: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 2);  unsqueeze_1889 = None
    unsqueeze_1891: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 3);  unsqueeze_1890 = None
    mul_977: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, unsqueeze_1891);  where_76 = unsqueeze_1891 = None
    mul_978: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, rsqrt_76);  sum_155 = rsqrt_76 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_977, relu_43, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_977 = primals_132 = None
    getitem_230: "f32[4, 320, 14, 14]" = convolution_backward_76[0]
    getitem_231: "f32[128, 320, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    le_77: "b8[4, 320, 14, 14]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_77: "f32[4, 320, 14, 14]" = torch.ops.aten.where.self(le_77, full_default, getitem_230);  le_77 = getitem_230 = None
    add_749: "f32[320]" = torch.ops.aten.add.Tensor(primals_495, 1e-05);  primals_495 = None
    rsqrt_77: "f32[320]" = torch.ops.aten.rsqrt.default(add_749);  add_749 = None
    unsqueeze_1892: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_494, 0);  primals_494 = None
    unsqueeze_1893: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, 2);  unsqueeze_1892 = None
    unsqueeze_1894: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1893, 3);  unsqueeze_1893 = None
    sum_156: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_198: "f32[4, 320, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_1894);  cat_19 = unsqueeze_1894 = None
    mul_979: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(where_77, sub_198);  sub_198 = None
    sum_157: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2, 3]);  mul_979 = None
    mul_984: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_77, primals_130);  primals_130 = None
    unsqueeze_1901: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_1902: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 2);  unsqueeze_1901 = None
    unsqueeze_1903: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 3);  unsqueeze_1902 = None
    mul_985: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(where_77, unsqueeze_1903);  where_77 = unsqueeze_1903 = None
    mul_986: "f32[320]" = torch.ops.aten.mul.Tensor(sum_157, rsqrt_77);  sum_157 = rsqrt_77 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_473: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_985, 1, 0, 256)
    slice_474: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_985, 1, 256, 288)
    slice_475: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_985, 1, 288, 320);  mul_985 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_750: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_744, slice_473);  add_744 = slice_473 = None
    add_751: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_745, slice_474);  add_745 = slice_474 = None
    add_752: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_746, slice_475);  add_746 = slice_475 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(add_752, relu_42, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_752 = primals_129 = None
    getitem_233: "f32[4, 128, 14, 14]" = convolution_backward_77[0]
    getitem_234: "f32[32, 128, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    le_78: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_78: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_78, full_default, getitem_233);  le_78 = getitem_233 = None
    add_753: "f32[128]" = torch.ops.aten.add.Tensor(primals_492, 1e-05);  primals_492 = None
    rsqrt_78: "f32[128]" = torch.ops.aten.rsqrt.default(add_753);  add_753 = None
    unsqueeze_1904: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_491, 0);  primals_491 = None
    unsqueeze_1905: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, 2);  unsqueeze_1904 = None
    unsqueeze_1906: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1905, 3);  unsqueeze_1905 = None
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_199: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1906);  convolution_41 = unsqueeze_1906 = None
    mul_987: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, sub_199);  sub_199 = None
    sum_159: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_992: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_78, primals_127);  primals_127 = None
    unsqueeze_1913: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_1914: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 2);  unsqueeze_1913 = None
    unsqueeze_1915: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, 3);  unsqueeze_1914 = None
    mul_993: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, unsqueeze_1915);  where_78 = unsqueeze_1915 = None
    mul_994: "f32[128]" = torch.ops.aten.mul.Tensor(sum_159, rsqrt_78);  sum_159 = rsqrt_78 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_993, relu_41, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_993 = primals_126 = None
    getitem_236: "f32[4, 288, 14, 14]" = convolution_backward_78[0]
    getitem_237: "f32[128, 288, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    le_79: "b8[4, 288, 14, 14]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_79: "f32[4, 288, 14, 14]" = torch.ops.aten.where.self(le_79, full_default, getitem_236);  le_79 = getitem_236 = None
    add_754: "f32[288]" = torch.ops.aten.add.Tensor(primals_489, 1e-05);  primals_489 = None
    rsqrt_79: "f32[288]" = torch.ops.aten.rsqrt.default(add_754);  add_754 = None
    unsqueeze_1916: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(primals_488, 0);  primals_488 = None
    unsqueeze_1917: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 2);  unsqueeze_1916 = None
    unsqueeze_1918: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1917, 3);  unsqueeze_1917 = None
    sum_160: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_200: "f32[4, 288, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_1918);  cat_18 = unsqueeze_1918 = None
    mul_995: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_79, sub_200);  sub_200 = None
    sum_161: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2, 3]);  mul_995 = None
    mul_1000: "f32[288]" = torch.ops.aten.mul.Tensor(rsqrt_79, primals_124);  primals_124 = None
    unsqueeze_1925: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_1926: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 2);  unsqueeze_1925 = None
    unsqueeze_1927: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, 3);  unsqueeze_1926 = None
    mul_1001: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_79, unsqueeze_1927);  where_79 = unsqueeze_1927 = None
    mul_1002: "f32[288]" = torch.ops.aten.mul.Tensor(sum_161, rsqrt_79);  sum_161 = rsqrt_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_476: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1001, 1, 0, 256)
    slice_477: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1001, 1, 256, 288);  mul_1001 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_755: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_750, slice_476);  add_750 = slice_476 = None
    add_756: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_751, slice_477);  add_751 = slice_477 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(add_756, relu_40, primals_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_756 = primals_123 = None
    getitem_239: "f32[4, 128, 14, 14]" = convolution_backward_79[0]
    getitem_240: "f32[32, 128, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    le_80: "b8[4, 128, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_80: "f32[4, 128, 14, 14]" = torch.ops.aten.where.self(le_80, full_default, getitem_239);  le_80 = getitem_239 = None
    add_757: "f32[128]" = torch.ops.aten.add.Tensor(primals_486, 1e-05);  primals_486 = None
    rsqrt_80: "f32[128]" = torch.ops.aten.rsqrt.default(add_757);  add_757 = None
    unsqueeze_1928: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_485, 0);  primals_485 = None
    unsqueeze_1929: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 2);  unsqueeze_1928 = None
    unsqueeze_1930: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1929, 3);  unsqueeze_1929 = None
    sum_162: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_201: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1930);  convolution_39 = unsqueeze_1930 = None
    mul_1003: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_80, sub_201);  sub_201 = None
    sum_163: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3]);  mul_1003 = None
    mul_1008: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_80, primals_121);  primals_121 = None
    unsqueeze_1937: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_1938: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1937, 2);  unsqueeze_1937 = None
    unsqueeze_1939: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, 3);  unsqueeze_1938 = None
    mul_1009: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_80, unsqueeze_1939);  where_80 = unsqueeze_1939 = None
    mul_1010: "f32[128]" = torch.ops.aten.mul.Tensor(sum_163, rsqrt_80);  sum_163 = rsqrt_80 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1009, relu_39, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1009 = primals_120 = None
    getitem_242: "f32[4, 256, 14, 14]" = convolution_backward_80[0]
    getitem_243: "f32[128, 256, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    le_81: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_81: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_81, full_default, getitem_242);  le_81 = getitem_242 = None
    add_758: "f32[256]" = torch.ops.aten.add.Tensor(primals_483, 1e-05);  primals_483 = None
    rsqrt_81: "f32[256]" = torch.ops.aten.rsqrt.default(add_758);  add_758 = None
    unsqueeze_1940: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_482, 0);  primals_482 = None
    unsqueeze_1941: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, 2);  unsqueeze_1940 = None
    unsqueeze_1942: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1941, 3);  unsqueeze_1941 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_202: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(avg_pool2d_1, unsqueeze_1942);  avg_pool2d_1 = unsqueeze_1942 = None
    mul_1011: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_81, sub_202);  sub_202 = None
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1011, [0, 2, 3]);  mul_1011 = None
    mul_1016: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_81, primals_118);  primals_118 = None
    unsqueeze_1949: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_1950: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1949, 2);  unsqueeze_1949 = None
    unsqueeze_1951: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 3);  unsqueeze_1950 = None
    mul_1017: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_81, unsqueeze_1951);  where_81 = unsqueeze_1951 = None
    mul_1018: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, rsqrt_81);  sum_165 = rsqrt_81 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_478: "f32[4, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1017, 1, 0, 256);  mul_1017 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_759: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_755, slice_478);  add_755 = slice_478 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    avg_pool2d_backward_1: "f32[4, 256, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(add_759, convolution_38, [2, 2], [2, 2], [0, 0], False, True, None);  add_759 = convolution_38 = None
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_1, relu_38, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_1 = primals_117 = None
    getitem_245: "f32[4, 512, 28, 28]" = convolution_backward_81[0]
    getitem_246: "f32[256, 512, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    le_82: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_82: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_82, full_default, getitem_245);  le_82 = getitem_245 = None
    add_760: "f32[512]" = torch.ops.aten.add.Tensor(primals_480, 1e-05);  primals_480 = None
    rsqrt_82: "f32[512]" = torch.ops.aten.rsqrt.default(add_760);  add_760 = None
    unsqueeze_1952: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_479, 0);  primals_479 = None
    unsqueeze_1953: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, 2);  unsqueeze_1952 = None
    unsqueeze_1954: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1953, 3);  unsqueeze_1953 = None
    sum_166: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_203: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_1954);  cat_17 = unsqueeze_1954 = None
    mul_1019: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_203);  sub_203 = None
    sum_167: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1019, [0, 2, 3]);  mul_1019 = None
    mul_1024: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_82, primals_115);  primals_115 = None
    unsqueeze_1961: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_1962: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1961, 2);  unsqueeze_1961 = None
    unsqueeze_1963: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 3);  unsqueeze_1962 = None
    mul_1025: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, unsqueeze_1963);  where_82 = unsqueeze_1963 = None
    mul_1026: "f32[512]" = torch.ops.aten.mul.Tensor(sum_167, rsqrt_82);  sum_167 = rsqrt_82 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    slice_479: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 0, 128)
    slice_480: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 128, 160)
    slice_481: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 160, 192)
    slice_482: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 192, 224)
    slice_483: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 224, 256)
    slice_484: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 256, 288)
    slice_485: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 288, 320)
    slice_486: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 320, 352)
    slice_487: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 352, 384)
    slice_488: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 384, 416)
    slice_489: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 416, 448)
    slice_490: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 448, 480)
    slice_491: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 480, 512);  mul_1025 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(slice_491, relu_37, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_491 = primals_114 = None
    getitem_248: "f32[4, 128, 28, 28]" = convolution_backward_82[0]
    getitem_249: "f32[32, 128, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    le_83: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_83: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_83, full_default, getitem_248);  le_83 = getitem_248 = None
    add_761: "f32[128]" = torch.ops.aten.add.Tensor(primals_477, 1e-05);  primals_477 = None
    rsqrt_83: "f32[128]" = torch.ops.aten.rsqrt.default(add_761);  add_761 = None
    unsqueeze_1964: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_476, 0);  primals_476 = None
    unsqueeze_1965: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, 2);  unsqueeze_1964 = None
    unsqueeze_1966: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1965, 3);  unsqueeze_1965 = None
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_204: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1966);  convolution_36 = unsqueeze_1966 = None
    mul_1027: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_204);  sub_204 = None
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
    mul_1032: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_83, primals_112);  primals_112 = None
    unsqueeze_1973: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_1974: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1973, 2);  unsqueeze_1973 = None
    unsqueeze_1975: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 3);  unsqueeze_1974 = None
    mul_1033: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, unsqueeze_1975);  where_83 = unsqueeze_1975 = None
    mul_1034: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, rsqrt_83);  sum_169 = rsqrt_83 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1033, relu_36, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1033 = primals_111 = None
    getitem_251: "f32[4, 480, 28, 28]" = convolution_backward_83[0]
    getitem_252: "f32[128, 480, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    le_84: "b8[4, 480, 28, 28]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_84: "f32[4, 480, 28, 28]" = torch.ops.aten.where.self(le_84, full_default, getitem_251);  le_84 = getitem_251 = None
    add_762: "f32[480]" = torch.ops.aten.add.Tensor(primals_474, 1e-05);  primals_474 = None
    rsqrt_84: "f32[480]" = torch.ops.aten.rsqrt.default(add_762);  add_762 = None
    unsqueeze_1976: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_473, 0);  primals_473 = None
    unsqueeze_1977: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, 2);  unsqueeze_1976 = None
    unsqueeze_1978: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1977, 3);  unsqueeze_1977 = None
    sum_170: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_205: "f32[4, 480, 28, 28]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_1978);  cat_16 = unsqueeze_1978 = None
    mul_1035: "f32[4, 480, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_205);  sub_205 = None
    sum_171: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_1035, [0, 2, 3]);  mul_1035 = None
    mul_1040: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_84, primals_109);  primals_109 = None
    unsqueeze_1985: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_1986: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1985, 2);  unsqueeze_1985 = None
    unsqueeze_1987: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 3);  unsqueeze_1986 = None
    mul_1041: "f32[4, 480, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, unsqueeze_1987);  where_84 = unsqueeze_1987 = None
    mul_1042: "f32[480]" = torch.ops.aten.mul.Tensor(sum_171, rsqrt_84);  sum_171 = rsqrt_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_492: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 0, 128)
    slice_493: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 128, 160)
    slice_494: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 160, 192)
    slice_495: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 192, 224)
    slice_496: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 224, 256)
    slice_497: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 256, 288)
    slice_498: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 288, 320)
    slice_499: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 320, 352)
    slice_500: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 352, 384)
    slice_501: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 384, 416)
    slice_502: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 416, 448)
    slice_503: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1041, 1, 448, 480);  mul_1041 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_763: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(slice_479, slice_492);  slice_479 = slice_492 = None
    add_764: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_480, slice_493);  slice_480 = slice_493 = None
    add_765: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_481, slice_494);  slice_481 = slice_494 = None
    add_766: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_482, slice_495);  slice_482 = slice_495 = None
    add_767: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_483, slice_496);  slice_483 = slice_496 = None
    add_768: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_484, slice_497);  slice_484 = slice_497 = None
    add_769: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_485, slice_498);  slice_485 = slice_498 = None
    add_770: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_486, slice_499);  slice_486 = slice_499 = None
    add_771: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_487, slice_500);  slice_487 = slice_500 = None
    add_772: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_488, slice_501);  slice_488 = slice_501 = None
    add_773: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_489, slice_502);  slice_489 = slice_502 = None
    add_774: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_490, slice_503);  slice_490 = slice_503 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(add_774, relu_35, primals_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_774 = primals_108 = None
    getitem_254: "f32[4, 128, 28, 28]" = convolution_backward_84[0]
    getitem_255: "f32[32, 128, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    le_85: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_85: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_85, full_default, getitem_254);  le_85 = getitem_254 = None
    add_775: "f32[128]" = torch.ops.aten.add.Tensor(primals_471, 1e-05);  primals_471 = None
    rsqrt_85: "f32[128]" = torch.ops.aten.rsqrt.default(add_775);  add_775 = None
    unsqueeze_1988: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_470, 0);  primals_470 = None
    unsqueeze_1989: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, 2);  unsqueeze_1988 = None
    unsqueeze_1990: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1989, 3);  unsqueeze_1989 = None
    sum_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_206: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1990);  convolution_34 = unsqueeze_1990 = None
    mul_1043: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_206);  sub_206 = None
    sum_173: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2, 3]);  mul_1043 = None
    mul_1048: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_85, primals_106);  primals_106 = None
    unsqueeze_1997: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_1998: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1997, 2);  unsqueeze_1997 = None
    unsqueeze_1999: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, 3);  unsqueeze_1998 = None
    mul_1049: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, unsqueeze_1999);  where_85 = unsqueeze_1999 = None
    mul_1050: "f32[128]" = torch.ops.aten.mul.Tensor(sum_173, rsqrt_85);  sum_173 = rsqrt_85 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1049, relu_34, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1049 = primals_105 = None
    getitem_257: "f32[4, 448, 28, 28]" = convolution_backward_85[0]
    getitem_258: "f32[128, 448, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    le_86: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_86: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_86, full_default, getitem_257);  le_86 = getitem_257 = None
    add_776: "f32[448]" = torch.ops.aten.add.Tensor(primals_468, 1e-05);  primals_468 = None
    rsqrt_86: "f32[448]" = torch.ops.aten.rsqrt.default(add_776);  add_776 = None
    unsqueeze_2000: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_467, 0);  primals_467 = None
    unsqueeze_2001: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, 2);  unsqueeze_2000 = None
    unsqueeze_2002: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2001, 3);  unsqueeze_2001 = None
    sum_174: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_207: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_2002);  cat_15 = unsqueeze_2002 = None
    mul_1051: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_207);  sub_207 = None
    sum_175: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_1051, [0, 2, 3]);  mul_1051 = None
    mul_1056: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_86, primals_103);  primals_103 = None
    unsqueeze_2009: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_2010: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2009, 2);  unsqueeze_2009 = None
    unsqueeze_2011: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 3);  unsqueeze_2010 = None
    mul_1057: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, unsqueeze_2011);  where_86 = unsqueeze_2011 = None
    mul_1058: "f32[448]" = torch.ops.aten.mul.Tensor(sum_175, rsqrt_86);  sum_175 = rsqrt_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_504: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 0, 128)
    slice_505: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 128, 160)
    slice_506: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 160, 192)
    slice_507: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 192, 224)
    slice_508: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 224, 256)
    slice_509: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 256, 288)
    slice_510: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 288, 320)
    slice_511: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 320, 352)
    slice_512: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 352, 384)
    slice_513: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 384, 416)
    slice_514: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1057, 1, 416, 448);  mul_1057 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_777: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_763, slice_504);  add_763 = slice_504 = None
    add_778: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_764, slice_505);  add_764 = slice_505 = None
    add_779: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_765, slice_506);  add_765 = slice_506 = None
    add_780: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_766, slice_507);  add_766 = slice_507 = None
    add_781: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_767, slice_508);  add_767 = slice_508 = None
    add_782: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_768, slice_509);  add_768 = slice_509 = None
    add_783: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_769, slice_510);  add_769 = slice_510 = None
    add_784: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_770, slice_511);  add_770 = slice_511 = None
    add_785: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_771, slice_512);  add_771 = slice_512 = None
    add_786: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_772, slice_513);  add_772 = slice_513 = None
    add_787: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_773, slice_514);  add_773 = slice_514 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(add_787, relu_33, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_787 = primals_102 = None
    getitem_260: "f32[4, 128, 28, 28]" = convolution_backward_86[0]
    getitem_261: "f32[32, 128, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None
    le_87: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_87: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_87, full_default, getitem_260);  le_87 = getitem_260 = None
    add_788: "f32[128]" = torch.ops.aten.add.Tensor(primals_465, 1e-05);  primals_465 = None
    rsqrt_87: "f32[128]" = torch.ops.aten.rsqrt.default(add_788);  add_788 = None
    unsqueeze_2012: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_464, 0);  primals_464 = None
    unsqueeze_2013: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, 2);  unsqueeze_2012 = None
    unsqueeze_2014: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2013, 3);  unsqueeze_2013 = None
    sum_176: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_208: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_2014);  convolution_32 = unsqueeze_2014 = None
    mul_1059: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_208);  sub_208 = None
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1064: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_87, primals_100);  primals_100 = None
    unsqueeze_2021: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_2022: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2021, 2);  unsqueeze_2021 = None
    unsqueeze_2023: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 3);  unsqueeze_2022 = None
    mul_1065: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, unsqueeze_2023);  where_87 = unsqueeze_2023 = None
    mul_1066: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, rsqrt_87);  sum_177 = rsqrt_87 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1065, relu_32, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1065 = primals_99 = None
    getitem_263: "f32[4, 416, 28, 28]" = convolution_backward_87[0]
    getitem_264: "f32[128, 416, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    le_88: "b8[4, 416, 28, 28]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_88: "f32[4, 416, 28, 28]" = torch.ops.aten.where.self(le_88, full_default, getitem_263);  le_88 = getitem_263 = None
    add_789: "f32[416]" = torch.ops.aten.add.Tensor(primals_462, 1e-05);  primals_462 = None
    rsqrt_88: "f32[416]" = torch.ops.aten.rsqrt.default(add_789);  add_789 = None
    unsqueeze_2024: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(primals_461, 0);  primals_461 = None
    unsqueeze_2025: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, 2);  unsqueeze_2024 = None
    unsqueeze_2026: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2025, 3);  unsqueeze_2025 = None
    sum_178: "f32[416]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_209: "f32[4, 416, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_2026);  cat_14 = unsqueeze_2026 = None
    mul_1067: "f32[4, 416, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_209);  sub_209 = None
    sum_179: "f32[416]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 2, 3]);  mul_1067 = None
    mul_1072: "f32[416]" = torch.ops.aten.mul.Tensor(rsqrt_88, primals_97);  primals_97 = None
    unsqueeze_2033: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_2034: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2033, 2);  unsqueeze_2033 = None
    unsqueeze_2035: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 3);  unsqueeze_2034 = None
    mul_1073: "f32[4, 416, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, unsqueeze_2035);  where_88 = unsqueeze_2035 = None
    mul_1074: "f32[416]" = torch.ops.aten.mul.Tensor(sum_179, rsqrt_88);  sum_179 = rsqrt_88 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_515: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 0, 128)
    slice_516: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 128, 160)
    slice_517: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 160, 192)
    slice_518: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 192, 224)
    slice_519: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 224, 256)
    slice_520: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 256, 288)
    slice_521: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 288, 320)
    slice_522: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 320, 352)
    slice_523: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 352, 384)
    slice_524: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1073, 1, 384, 416);  mul_1073 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_790: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_777, slice_515);  add_777 = slice_515 = None
    add_791: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_778, slice_516);  add_778 = slice_516 = None
    add_792: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_779, slice_517);  add_779 = slice_517 = None
    add_793: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_780, slice_518);  add_780 = slice_518 = None
    add_794: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_781, slice_519);  add_781 = slice_519 = None
    add_795: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_782, slice_520);  add_782 = slice_520 = None
    add_796: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_783, slice_521);  add_783 = slice_521 = None
    add_797: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_784, slice_522);  add_784 = slice_522 = None
    add_798: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_785, slice_523);  add_785 = slice_523 = None
    add_799: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_786, slice_524);  add_786 = slice_524 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(add_799, relu_31, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_799 = primals_96 = None
    getitem_266: "f32[4, 128, 28, 28]" = convolution_backward_88[0]
    getitem_267: "f32[32, 128, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    le_89: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_89: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_89, full_default, getitem_266);  le_89 = getitem_266 = None
    add_800: "f32[128]" = torch.ops.aten.add.Tensor(primals_459, 1e-05);  primals_459 = None
    rsqrt_89: "f32[128]" = torch.ops.aten.rsqrt.default(add_800);  add_800 = None
    unsqueeze_2036: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_458, 0);  primals_458 = None
    unsqueeze_2037: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, 2);  unsqueeze_2036 = None
    unsqueeze_2038: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2037, 3);  unsqueeze_2037 = None
    sum_180: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_210: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_2038);  convolution_30 = unsqueeze_2038 = None
    mul_1075: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, sub_210);  sub_210 = None
    sum_181: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1075, [0, 2, 3]);  mul_1075 = None
    mul_1080: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_89, primals_94);  primals_94 = None
    unsqueeze_2045: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
    unsqueeze_2046: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2045, 2);  unsqueeze_2045 = None
    unsqueeze_2047: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, 3);  unsqueeze_2046 = None
    mul_1081: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, unsqueeze_2047);  where_89 = unsqueeze_2047 = None
    mul_1082: "f32[128]" = torch.ops.aten.mul.Tensor(sum_181, rsqrt_89);  sum_181 = rsqrt_89 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1081, relu_30, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1081 = primals_93 = None
    getitem_269: "f32[4, 384, 28, 28]" = convolution_backward_89[0]
    getitem_270: "f32[128, 384, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    le_90: "b8[4, 384, 28, 28]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_90: "f32[4, 384, 28, 28]" = torch.ops.aten.where.self(le_90, full_default, getitem_269);  le_90 = getitem_269 = None
    add_801: "f32[384]" = torch.ops.aten.add.Tensor(primals_456, 1e-05);  primals_456 = None
    rsqrt_90: "f32[384]" = torch.ops.aten.rsqrt.default(add_801);  add_801 = None
    unsqueeze_2048: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(primals_455, 0);  primals_455 = None
    unsqueeze_2049: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, 2);  unsqueeze_2048 = None
    unsqueeze_2050: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2049, 3);  unsqueeze_2049 = None
    sum_182: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_211: "f32[4, 384, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_2050);  cat_13 = unsqueeze_2050 = None
    mul_1083: "f32[4, 384, 28, 28]" = torch.ops.aten.mul.Tensor(where_90, sub_211);  sub_211 = None
    sum_183: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1083, [0, 2, 3]);  mul_1083 = None
    mul_1088: "f32[384]" = torch.ops.aten.mul.Tensor(rsqrt_90, primals_91);  primals_91 = None
    unsqueeze_2057: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
    unsqueeze_2058: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2057, 2);  unsqueeze_2057 = None
    unsqueeze_2059: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 3);  unsqueeze_2058 = None
    mul_1089: "f32[4, 384, 28, 28]" = torch.ops.aten.mul.Tensor(where_90, unsqueeze_2059);  where_90 = unsqueeze_2059 = None
    mul_1090: "f32[384]" = torch.ops.aten.mul.Tensor(sum_183, rsqrt_90);  sum_183 = rsqrt_90 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_525: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 0, 128)
    slice_526: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 128, 160)
    slice_527: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 160, 192)
    slice_528: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 192, 224)
    slice_529: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 224, 256)
    slice_530: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 256, 288)
    slice_531: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 288, 320)
    slice_532: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 320, 352)
    slice_533: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1089, 1, 352, 384);  mul_1089 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_802: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_790, slice_525);  add_790 = slice_525 = None
    add_803: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_791, slice_526);  add_791 = slice_526 = None
    add_804: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_792, slice_527);  add_792 = slice_527 = None
    add_805: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_793, slice_528);  add_793 = slice_528 = None
    add_806: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_794, slice_529);  add_794 = slice_529 = None
    add_807: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_795, slice_530);  add_795 = slice_530 = None
    add_808: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_796, slice_531);  add_796 = slice_531 = None
    add_809: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_797, slice_532);  add_797 = slice_532 = None
    add_810: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_798, slice_533);  add_798 = slice_533 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(add_810, relu_29, primals_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_810 = primals_90 = None
    getitem_272: "f32[4, 128, 28, 28]" = convolution_backward_90[0]
    getitem_273: "f32[32, 128, 3, 3]" = convolution_backward_90[1];  convolution_backward_90 = None
    le_91: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_91: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_91, full_default, getitem_272);  le_91 = getitem_272 = None
    add_811: "f32[128]" = torch.ops.aten.add.Tensor(primals_453, 1e-05);  primals_453 = None
    rsqrt_91: "f32[128]" = torch.ops.aten.rsqrt.default(add_811);  add_811 = None
    unsqueeze_2060: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_452, 0);  primals_452 = None
    unsqueeze_2061: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, 2);  unsqueeze_2060 = None
    unsqueeze_2062: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2061, 3);  unsqueeze_2061 = None
    sum_184: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_212: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_2062);  convolution_28 = unsqueeze_2062 = None
    mul_1091: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_91, sub_212);  sub_212 = None
    sum_185: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1091, [0, 2, 3]);  mul_1091 = None
    mul_1096: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_91, primals_88);  primals_88 = None
    unsqueeze_2069: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_2070: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2069, 2);  unsqueeze_2069 = None
    unsqueeze_2071: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, 3);  unsqueeze_2070 = None
    mul_1097: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_91, unsqueeze_2071);  where_91 = unsqueeze_2071 = None
    mul_1098: "f32[128]" = torch.ops.aten.mul.Tensor(sum_185, rsqrt_91);  sum_185 = rsqrt_91 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1097, relu_28, primals_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1097 = primals_87 = None
    getitem_275: "f32[4, 352, 28, 28]" = convolution_backward_91[0]
    getitem_276: "f32[128, 352, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    le_92: "b8[4, 352, 28, 28]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_92: "f32[4, 352, 28, 28]" = torch.ops.aten.where.self(le_92, full_default, getitem_275);  le_92 = getitem_275 = None
    add_812: "f32[352]" = torch.ops.aten.add.Tensor(primals_450, 1e-05);  primals_450 = None
    rsqrt_92: "f32[352]" = torch.ops.aten.rsqrt.default(add_812);  add_812 = None
    unsqueeze_2072: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(primals_449, 0);  primals_449 = None
    unsqueeze_2073: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, 2);  unsqueeze_2072 = None
    unsqueeze_2074: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2073, 3);  unsqueeze_2073 = None
    sum_186: "f32[352]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_213: "f32[4, 352, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_2074);  cat_12 = unsqueeze_2074 = None
    mul_1099: "f32[4, 352, 28, 28]" = torch.ops.aten.mul.Tensor(where_92, sub_213);  sub_213 = None
    sum_187: "f32[352]" = torch.ops.aten.sum.dim_IntList(mul_1099, [0, 2, 3]);  mul_1099 = None
    mul_1104: "f32[352]" = torch.ops.aten.mul.Tensor(rsqrt_92, primals_85);  primals_85 = None
    unsqueeze_2081: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
    unsqueeze_2082: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2081, 2);  unsqueeze_2081 = None
    unsqueeze_2083: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, 3);  unsqueeze_2082 = None
    mul_1105: "f32[4, 352, 28, 28]" = torch.ops.aten.mul.Tensor(where_92, unsqueeze_2083);  where_92 = unsqueeze_2083 = None
    mul_1106: "f32[352]" = torch.ops.aten.mul.Tensor(sum_187, rsqrt_92);  sum_187 = rsqrt_92 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_534: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 0, 128)
    slice_535: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 128, 160)
    slice_536: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 160, 192)
    slice_537: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 192, 224)
    slice_538: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 224, 256)
    slice_539: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 256, 288)
    slice_540: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 288, 320)
    slice_541: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 320, 352);  mul_1105 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_813: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_802, slice_534);  add_802 = slice_534 = None
    add_814: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_803, slice_535);  add_803 = slice_535 = None
    add_815: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_804, slice_536);  add_804 = slice_536 = None
    add_816: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_805, slice_537);  add_805 = slice_537 = None
    add_817: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_806, slice_538);  add_806 = slice_538 = None
    add_818: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_807, slice_539);  add_807 = slice_539 = None
    add_819: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_808, slice_540);  add_808 = slice_540 = None
    add_820: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_809, slice_541);  add_809 = slice_541 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(add_820, relu_27, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_820 = primals_84 = None
    getitem_278: "f32[4, 128, 28, 28]" = convolution_backward_92[0]
    getitem_279: "f32[32, 128, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    le_93: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_93: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_93, full_default, getitem_278);  le_93 = getitem_278 = None
    add_821: "f32[128]" = torch.ops.aten.add.Tensor(primals_447, 1e-05);  primals_447 = None
    rsqrt_93: "f32[128]" = torch.ops.aten.rsqrt.default(add_821);  add_821 = None
    unsqueeze_2084: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_446, 0);  primals_446 = None
    unsqueeze_2085: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, 2);  unsqueeze_2084 = None
    unsqueeze_2086: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2085, 3);  unsqueeze_2085 = None
    sum_188: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_214: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_2086);  convolution_26 = unsqueeze_2086 = None
    mul_1107: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_93, sub_214);  sub_214 = None
    sum_189: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3]);  mul_1107 = None
    mul_1112: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_93, primals_82);  primals_82 = None
    unsqueeze_2093: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_2094: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2093, 2);  unsqueeze_2093 = None
    unsqueeze_2095: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, 3);  unsqueeze_2094 = None
    mul_1113: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_93, unsqueeze_2095);  where_93 = unsqueeze_2095 = None
    mul_1114: "f32[128]" = torch.ops.aten.mul.Tensor(sum_189, rsqrt_93);  sum_189 = rsqrt_93 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1113, relu_26, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1113 = primals_81 = None
    getitem_281: "f32[4, 320, 28, 28]" = convolution_backward_93[0]
    getitem_282: "f32[128, 320, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    le_94: "b8[4, 320, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_94: "f32[4, 320, 28, 28]" = torch.ops.aten.where.self(le_94, full_default, getitem_281);  le_94 = getitem_281 = None
    add_822: "f32[320]" = torch.ops.aten.add.Tensor(primals_444, 1e-05);  primals_444 = None
    rsqrt_94: "f32[320]" = torch.ops.aten.rsqrt.default(add_822);  add_822 = None
    unsqueeze_2096: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(primals_443, 0);  primals_443 = None
    unsqueeze_2097: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, 2);  unsqueeze_2096 = None
    unsqueeze_2098: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2097, 3);  unsqueeze_2097 = None
    sum_190: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_215: "f32[4, 320, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_2098);  cat_11 = unsqueeze_2098 = None
    mul_1115: "f32[4, 320, 28, 28]" = torch.ops.aten.mul.Tensor(where_94, sub_215);  sub_215 = None
    sum_191: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3]);  mul_1115 = None
    mul_1120: "f32[320]" = torch.ops.aten.mul.Tensor(rsqrt_94, primals_79);  primals_79 = None
    unsqueeze_2105: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_2106: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2105, 2);  unsqueeze_2105 = None
    unsqueeze_2107: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 3);  unsqueeze_2106 = None
    mul_1121: "f32[4, 320, 28, 28]" = torch.ops.aten.mul.Tensor(where_94, unsqueeze_2107);  where_94 = unsqueeze_2107 = None
    mul_1122: "f32[320]" = torch.ops.aten.mul.Tensor(sum_191, rsqrt_94);  sum_191 = rsqrt_94 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_542: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 0, 128)
    slice_543: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 128, 160)
    slice_544: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 160, 192)
    slice_545: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 192, 224)
    slice_546: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 224, 256)
    slice_547: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 256, 288)
    slice_548: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1121, 1, 288, 320);  mul_1121 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_823: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_813, slice_542);  add_813 = slice_542 = None
    add_824: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_814, slice_543);  add_814 = slice_543 = None
    add_825: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_815, slice_544);  add_815 = slice_544 = None
    add_826: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_816, slice_545);  add_816 = slice_545 = None
    add_827: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_817, slice_546);  add_817 = slice_546 = None
    add_828: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_818, slice_547);  add_818 = slice_547 = None
    add_829: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_819, slice_548);  add_819 = slice_548 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(add_829, relu_25, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_829 = primals_78 = None
    getitem_284: "f32[4, 128, 28, 28]" = convolution_backward_94[0]
    getitem_285: "f32[32, 128, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    le_95: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_95: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_95, full_default, getitem_284);  le_95 = getitem_284 = None
    add_830: "f32[128]" = torch.ops.aten.add.Tensor(primals_441, 1e-05);  primals_441 = None
    rsqrt_95: "f32[128]" = torch.ops.aten.rsqrt.default(add_830);  add_830 = None
    unsqueeze_2108: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_440, 0);  primals_440 = None
    unsqueeze_2109: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, 2);  unsqueeze_2108 = None
    unsqueeze_2110: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2109, 3);  unsqueeze_2109 = None
    sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_216: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_2110);  convolution_24 = unsqueeze_2110 = None
    mul_1123: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_95, sub_216);  sub_216 = None
    sum_193: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1123, [0, 2, 3]);  mul_1123 = None
    mul_1128: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_95, primals_76);  primals_76 = None
    unsqueeze_2117: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_2118: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2117, 2);  unsqueeze_2117 = None
    unsqueeze_2119: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 3);  unsqueeze_2118 = None
    mul_1129: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_95, unsqueeze_2119);  where_95 = unsqueeze_2119 = None
    mul_1130: "f32[128]" = torch.ops.aten.mul.Tensor(sum_193, rsqrt_95);  sum_193 = rsqrt_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1129, relu_24, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1129 = primals_75 = None
    getitem_287: "f32[4, 288, 28, 28]" = convolution_backward_95[0]
    getitem_288: "f32[128, 288, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    le_96: "b8[4, 288, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_96: "f32[4, 288, 28, 28]" = torch.ops.aten.where.self(le_96, full_default, getitem_287);  le_96 = getitem_287 = None
    add_831: "f32[288]" = torch.ops.aten.add.Tensor(primals_438, 1e-05);  primals_438 = None
    rsqrt_96: "f32[288]" = torch.ops.aten.rsqrt.default(add_831);  add_831 = None
    unsqueeze_2120: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(primals_437, 0);  primals_437 = None
    unsqueeze_2121: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, 2);  unsqueeze_2120 = None
    unsqueeze_2122: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2121, 3);  unsqueeze_2121 = None
    sum_194: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_217: "f32[4, 288, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_2122);  cat_10 = unsqueeze_2122 = None
    mul_1131: "f32[4, 288, 28, 28]" = torch.ops.aten.mul.Tensor(where_96, sub_217);  sub_217 = None
    sum_195: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_1131, [0, 2, 3]);  mul_1131 = None
    mul_1136: "f32[288]" = torch.ops.aten.mul.Tensor(rsqrt_96, primals_73);  primals_73 = None
    unsqueeze_2129: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_2130: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2129, 2);  unsqueeze_2129 = None
    unsqueeze_2131: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 3);  unsqueeze_2130 = None
    mul_1137: "f32[4, 288, 28, 28]" = torch.ops.aten.mul.Tensor(where_96, unsqueeze_2131);  where_96 = unsqueeze_2131 = None
    mul_1138: "f32[288]" = torch.ops.aten.mul.Tensor(sum_195, rsqrt_96);  sum_195 = rsqrt_96 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_549: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 0, 128)
    slice_550: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 128, 160)
    slice_551: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 160, 192)
    slice_552: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 192, 224)
    slice_553: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 224, 256)
    slice_554: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1137, 1, 256, 288);  mul_1137 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_832: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_823, slice_549);  add_823 = slice_549 = None
    add_833: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_824, slice_550);  add_824 = slice_550 = None
    add_834: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_825, slice_551);  add_825 = slice_551 = None
    add_835: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_826, slice_552);  add_826 = slice_552 = None
    add_836: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_827, slice_553);  add_827 = slice_553 = None
    add_837: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_828, slice_554);  add_828 = slice_554 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(add_837, relu_23, primals_72, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_837 = primals_72 = None
    getitem_290: "f32[4, 128, 28, 28]" = convolution_backward_96[0]
    getitem_291: "f32[32, 128, 3, 3]" = convolution_backward_96[1];  convolution_backward_96 = None
    le_97: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_97: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_97, full_default, getitem_290);  le_97 = getitem_290 = None
    add_838: "f32[128]" = torch.ops.aten.add.Tensor(primals_435, 1e-05);  primals_435 = None
    rsqrt_97: "f32[128]" = torch.ops.aten.rsqrt.default(add_838);  add_838 = None
    unsqueeze_2132: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_434, 0);  primals_434 = None
    unsqueeze_2133: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, 2);  unsqueeze_2132 = None
    unsqueeze_2134: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2133, 3);  unsqueeze_2133 = None
    sum_196: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_218: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_2134);  convolution_22 = unsqueeze_2134 = None
    mul_1139: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_97, sub_218);  sub_218 = None
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1139, [0, 2, 3]);  mul_1139 = None
    mul_1144: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_97, primals_70);  primals_70 = None
    unsqueeze_2141: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1144, 0);  mul_1144 = None
    unsqueeze_2142: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2141, 2);  unsqueeze_2141 = None
    unsqueeze_2143: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 3);  unsqueeze_2142 = None
    mul_1145: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_97, unsqueeze_2143);  where_97 = unsqueeze_2143 = None
    mul_1146: "f32[128]" = torch.ops.aten.mul.Tensor(sum_197, rsqrt_97);  sum_197 = rsqrt_97 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1145, relu_22, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1145 = primals_69 = None
    getitem_293: "f32[4, 256, 28, 28]" = convolution_backward_97[0]
    getitem_294: "f32[128, 256, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
    le_98: "b8[4, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_98: "f32[4, 256, 28, 28]" = torch.ops.aten.where.self(le_98, full_default, getitem_293);  le_98 = getitem_293 = None
    add_839: "f32[256]" = torch.ops.aten.add.Tensor(primals_432, 1e-05);  primals_432 = None
    rsqrt_98: "f32[256]" = torch.ops.aten.rsqrt.default(add_839);  add_839 = None
    unsqueeze_2144: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_431, 0);  primals_431 = None
    unsqueeze_2145: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, 2);  unsqueeze_2144 = None
    unsqueeze_2146: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2145, 3);  unsqueeze_2145 = None
    sum_198: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_219: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_2146);  cat_9 = unsqueeze_2146 = None
    mul_1147: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_98, sub_219);  sub_219 = None
    sum_199: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3]);  mul_1147 = None
    mul_1152: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_98, primals_67);  primals_67 = None
    unsqueeze_2153: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_2154: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2153, 2);  unsqueeze_2153 = None
    unsqueeze_2155: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, 3);  unsqueeze_2154 = None
    mul_1153: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_98, unsqueeze_2155);  where_98 = unsqueeze_2155 = None
    mul_1154: "f32[256]" = torch.ops.aten.mul.Tensor(sum_199, rsqrt_98);  sum_199 = rsqrt_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_555: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 0, 128)
    slice_556: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 128, 160)
    slice_557: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 160, 192)
    slice_558: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 192, 224)
    slice_559: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 224, 256);  mul_1153 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_840: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_832, slice_555);  add_832 = slice_555 = None
    add_841: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_833, slice_556);  add_833 = slice_556 = None
    add_842: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_834, slice_557);  add_834 = slice_557 = None
    add_843: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_835, slice_558);  add_835 = slice_558 = None
    add_844: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_836, slice_559);  add_836 = slice_559 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(add_844, relu_21, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_844 = primals_66 = None
    getitem_296: "f32[4, 128, 28, 28]" = convolution_backward_98[0]
    getitem_297: "f32[32, 128, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    le_99: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_99: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_99, full_default, getitem_296);  le_99 = getitem_296 = None
    add_845: "f32[128]" = torch.ops.aten.add.Tensor(primals_429, 1e-05);  primals_429 = None
    rsqrt_99: "f32[128]" = torch.ops.aten.rsqrt.default(add_845);  add_845 = None
    unsqueeze_2156: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_428, 0);  primals_428 = None
    unsqueeze_2157: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, 2);  unsqueeze_2156 = None
    unsqueeze_2158: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2157, 3);  unsqueeze_2157 = None
    sum_200: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_220: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_2158);  convolution_20 = unsqueeze_2158 = None
    mul_1155: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_99, sub_220);  sub_220 = None
    sum_201: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1155, [0, 2, 3]);  mul_1155 = None
    mul_1160: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_99, primals_64);  primals_64 = None
    unsqueeze_2165: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_2166: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2165, 2);  unsqueeze_2165 = None
    unsqueeze_2167: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, 3);  unsqueeze_2166 = None
    mul_1161: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_99, unsqueeze_2167);  where_99 = unsqueeze_2167 = None
    mul_1162: "f32[128]" = torch.ops.aten.mul.Tensor(sum_201, rsqrt_99);  sum_201 = rsqrt_99 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1161, relu_20, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1161 = primals_63 = None
    getitem_299: "f32[4, 224, 28, 28]" = convolution_backward_99[0]
    getitem_300: "f32[128, 224, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    le_100: "b8[4, 224, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_100: "f32[4, 224, 28, 28]" = torch.ops.aten.where.self(le_100, full_default, getitem_299);  le_100 = getitem_299 = None
    add_846: "f32[224]" = torch.ops.aten.add.Tensor(primals_426, 1e-05);  primals_426 = None
    rsqrt_100: "f32[224]" = torch.ops.aten.rsqrt.default(add_846);  add_846 = None
    unsqueeze_2168: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_425, 0);  primals_425 = None
    unsqueeze_2169: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, 2);  unsqueeze_2168 = None
    unsqueeze_2170: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2169, 3);  unsqueeze_2169 = None
    sum_202: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_221: "f32[4, 224, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_2170);  cat_8 = unsqueeze_2170 = None
    mul_1163: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_100, sub_221);  sub_221 = None
    sum_203: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1163, [0, 2, 3]);  mul_1163 = None
    mul_1168: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_100, primals_61);  primals_61 = None
    unsqueeze_2177: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_2178: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2177, 2);  unsqueeze_2177 = None
    unsqueeze_2179: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 3);  unsqueeze_2178 = None
    mul_1169: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_100, unsqueeze_2179);  where_100 = unsqueeze_2179 = None
    mul_1170: "f32[224]" = torch.ops.aten.mul.Tensor(sum_203, rsqrt_100);  sum_203 = rsqrt_100 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_560: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 0, 128)
    slice_561: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 128, 160)
    slice_562: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 160, 192)
    slice_563: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 192, 224);  mul_1169 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_847: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_840, slice_560);  add_840 = slice_560 = None
    add_848: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_841, slice_561);  add_841 = slice_561 = None
    add_849: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_842, slice_562);  add_842 = slice_562 = None
    add_850: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_843, slice_563);  add_843 = slice_563 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(add_850, relu_19, primals_60, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_850 = primals_60 = None
    getitem_302: "f32[4, 128, 28, 28]" = convolution_backward_100[0]
    getitem_303: "f32[32, 128, 3, 3]" = convolution_backward_100[1];  convolution_backward_100 = None
    le_101: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_101: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_101, full_default, getitem_302);  le_101 = getitem_302 = None
    add_851: "f32[128]" = torch.ops.aten.add.Tensor(primals_423, 1e-05);  primals_423 = None
    rsqrt_101: "f32[128]" = torch.ops.aten.rsqrt.default(add_851);  add_851 = None
    unsqueeze_2180: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_422, 0);  primals_422 = None
    unsqueeze_2181: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, 2);  unsqueeze_2180 = None
    unsqueeze_2182: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2181, 3);  unsqueeze_2181 = None
    sum_204: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_222: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_2182);  convolution_18 = unsqueeze_2182 = None
    mul_1171: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_101, sub_222);  sub_222 = None
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3]);  mul_1171 = None
    mul_1176: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_101, primals_58);  primals_58 = None
    unsqueeze_2189: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
    unsqueeze_2190: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2189, 2);  unsqueeze_2189 = None
    unsqueeze_2191: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 3);  unsqueeze_2190 = None
    mul_1177: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_101, unsqueeze_2191);  where_101 = unsqueeze_2191 = None
    mul_1178: "f32[128]" = torch.ops.aten.mul.Tensor(sum_205, rsqrt_101);  sum_205 = rsqrt_101 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1177, relu_18, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1177 = primals_57 = None
    getitem_305: "f32[4, 192, 28, 28]" = convolution_backward_101[0]
    getitem_306: "f32[128, 192, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    le_102: "b8[4, 192, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_102: "f32[4, 192, 28, 28]" = torch.ops.aten.where.self(le_102, full_default, getitem_305);  le_102 = getitem_305 = None
    add_852: "f32[192]" = torch.ops.aten.add.Tensor(primals_420, 1e-05);  primals_420 = None
    rsqrt_102: "f32[192]" = torch.ops.aten.rsqrt.default(add_852);  add_852 = None
    unsqueeze_2192: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_419, 0);  primals_419 = None
    unsqueeze_2193: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, 2);  unsqueeze_2192 = None
    unsqueeze_2194: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2193, 3);  unsqueeze_2193 = None
    sum_206: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_223: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_2194);  cat_7 = unsqueeze_2194 = None
    mul_1179: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_102, sub_223);  sub_223 = None
    sum_207: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1179, [0, 2, 3]);  mul_1179 = None
    mul_1184: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_102, primals_55);  primals_55 = None
    unsqueeze_2201: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
    unsqueeze_2202: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2201, 2);  unsqueeze_2201 = None
    unsqueeze_2203: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 3);  unsqueeze_2202 = None
    mul_1185: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_102, unsqueeze_2203);  where_102 = unsqueeze_2203 = None
    mul_1186: "f32[192]" = torch.ops.aten.mul.Tensor(sum_207, rsqrt_102);  sum_207 = rsqrt_102 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_564: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1185, 1, 0, 128)
    slice_565: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1185, 1, 128, 160)
    slice_566: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1185, 1, 160, 192);  mul_1185 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_853: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_847, slice_564);  add_847 = slice_564 = None
    add_854: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_848, slice_565);  add_848 = slice_565 = None
    add_855: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_849, slice_566);  add_849 = slice_566 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(add_855, relu_17, primals_54, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_855 = primals_54 = None
    getitem_308: "f32[4, 128, 28, 28]" = convolution_backward_102[0]
    getitem_309: "f32[32, 128, 3, 3]" = convolution_backward_102[1];  convolution_backward_102 = None
    le_103: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_103: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_103, full_default, getitem_308);  le_103 = getitem_308 = None
    add_856: "f32[128]" = torch.ops.aten.add.Tensor(primals_417, 1e-05);  primals_417 = None
    rsqrt_103: "f32[128]" = torch.ops.aten.rsqrt.default(add_856);  add_856 = None
    unsqueeze_2204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_416, 0);  primals_416 = None
    unsqueeze_2205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, 2);  unsqueeze_2204 = None
    unsqueeze_2206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2205, 3);  unsqueeze_2205 = None
    sum_208: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_224: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_2206);  convolution_16 = unsqueeze_2206 = None
    mul_1187: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_103, sub_224);  sub_224 = None
    sum_209: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1192: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_103, primals_52);  primals_52 = None
    unsqueeze_2213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_2214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2213, 2);  unsqueeze_2213 = None
    unsqueeze_2215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 3);  unsqueeze_2214 = None
    mul_1193: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_103, unsqueeze_2215);  where_103 = unsqueeze_2215 = None
    mul_1194: "f32[128]" = torch.ops.aten.mul.Tensor(sum_209, rsqrt_103);  sum_209 = rsqrt_103 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1193, relu_16, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1193 = primals_51 = None
    getitem_311: "f32[4, 160, 28, 28]" = convolution_backward_103[0]
    getitem_312: "f32[128, 160, 1, 1]" = convolution_backward_103[1];  convolution_backward_103 = None
    le_104: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_104: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_104, full_default, getitem_311);  le_104 = getitem_311 = None
    add_857: "f32[160]" = torch.ops.aten.add.Tensor(primals_414, 1e-05);  primals_414 = None
    rsqrt_104: "f32[160]" = torch.ops.aten.rsqrt.default(add_857);  add_857 = None
    unsqueeze_2216: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_413, 0);  primals_413 = None
    unsqueeze_2217: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, 2);  unsqueeze_2216 = None
    unsqueeze_2218: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2217, 3);  unsqueeze_2217 = None
    sum_210: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_225: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_2218);  cat_6 = unsqueeze_2218 = None
    mul_1195: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_104, sub_225);  sub_225 = None
    sum_211: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1195, [0, 2, 3]);  mul_1195 = None
    mul_1200: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_104, primals_49);  primals_49 = None
    unsqueeze_2225: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
    unsqueeze_2226: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2225, 2);  unsqueeze_2225 = None
    unsqueeze_2227: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 3);  unsqueeze_2226 = None
    mul_1201: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_104, unsqueeze_2227);  where_104 = unsqueeze_2227 = None
    mul_1202: "f32[160]" = torch.ops.aten.mul.Tensor(sum_211, rsqrt_104);  sum_211 = rsqrt_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_567: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1201, 1, 0, 128)
    slice_568: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1201, 1, 128, 160);  mul_1201 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_858: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_853, slice_567);  add_853 = slice_567 = None
    add_859: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_854, slice_568);  add_854 = slice_568 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(add_859, relu_15, primals_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_859 = primals_48 = None
    getitem_314: "f32[4, 128, 28, 28]" = convolution_backward_104[0]
    getitem_315: "f32[32, 128, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    le_105: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_105: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_105, full_default, getitem_314);  le_105 = getitem_314 = None
    add_860: "f32[128]" = torch.ops.aten.add.Tensor(primals_411, 1e-05);  primals_411 = None
    rsqrt_105: "f32[128]" = torch.ops.aten.rsqrt.default(add_860);  add_860 = None
    unsqueeze_2228: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_410, 0);  primals_410 = None
    unsqueeze_2229: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2228, 2);  unsqueeze_2228 = None
    unsqueeze_2230: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2229, 3);  unsqueeze_2229 = None
    sum_212: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_226: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_2230);  convolution_14 = unsqueeze_2230 = None
    mul_1203: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_105, sub_226);  sub_226 = None
    sum_213: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1203, [0, 2, 3]);  mul_1203 = None
    mul_1208: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_105, primals_46);  primals_46 = None
    unsqueeze_2237: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
    unsqueeze_2238: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2237, 2);  unsqueeze_2237 = None
    unsqueeze_2239: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, 3);  unsqueeze_2238 = None
    mul_1209: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_105, unsqueeze_2239);  where_105 = unsqueeze_2239 = None
    mul_1210: "f32[128]" = torch.ops.aten.mul.Tensor(sum_213, rsqrt_105);  sum_213 = rsqrt_105 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1209, relu_14, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1209 = primals_45 = None
    getitem_317: "f32[4, 128, 28, 28]" = convolution_backward_105[0]
    getitem_318: "f32[128, 128, 1, 1]" = convolution_backward_105[1];  convolution_backward_105 = None
    le_106: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_106: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_106, full_default, getitem_317);  le_106 = getitem_317 = None
    add_861: "f32[128]" = torch.ops.aten.add.Tensor(primals_408, 1e-05);  primals_408 = None
    rsqrt_106: "f32[128]" = torch.ops.aten.rsqrt.default(add_861);  add_861 = None
    unsqueeze_2240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_407, 0);  primals_407 = None
    unsqueeze_2241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, 2);  unsqueeze_2240 = None
    unsqueeze_2242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2241, 3);  unsqueeze_2241 = None
    sum_214: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_227: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_2242);  avg_pool2d = unsqueeze_2242 = None
    mul_1211: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_106, sub_227);  sub_227 = None
    sum_215: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1211, [0, 2, 3]);  mul_1211 = None
    mul_1216: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_106, primals_43);  primals_43 = None
    unsqueeze_2249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_2250: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2249, 2);  unsqueeze_2249 = None
    unsqueeze_2251: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, 3);  unsqueeze_2250 = None
    mul_1217: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_106, unsqueeze_2251);  where_106 = unsqueeze_2251 = None
    mul_1218: "f32[128]" = torch.ops.aten.mul.Tensor(sum_215, rsqrt_106);  sum_215 = rsqrt_106 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_569: "f32[4, 128, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1217, 1, 0, 128);  mul_1217 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_862: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_858, slice_569);  add_858 = slice_569 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    avg_pool2d_backward_2: "f32[4, 128, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(add_862, convolution_13, [2, 2], [2, 2], [0, 0], False, True, None);  add_862 = convolution_13 = None
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_2, relu_13, primals_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_2 = primals_42 = None
    getitem_320: "f32[4, 256, 56, 56]" = convolution_backward_106[0]
    getitem_321: "f32[128, 256, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    le_107: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_107: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_107, full_default, getitem_320);  le_107 = getitem_320 = None
    add_863: "f32[256]" = torch.ops.aten.add.Tensor(primals_405, 1e-05);  primals_405 = None
    rsqrt_107: "f32[256]" = torch.ops.aten.rsqrt.default(add_863);  add_863 = None
    unsqueeze_2252: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_404, 0);  primals_404 = None
    unsqueeze_2253: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, 2);  unsqueeze_2252 = None
    unsqueeze_2254: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2253, 3);  unsqueeze_2253 = None
    sum_216: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_228: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_2254);  cat_5 = unsqueeze_2254 = None
    mul_1219: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_107, sub_228);  sub_228 = None
    sum_217: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1219, [0, 2, 3]);  mul_1219 = None
    mul_1224: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_107, primals_40);  primals_40 = None
    unsqueeze_2261: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
    unsqueeze_2262: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2261, 2);  unsqueeze_2261 = None
    unsqueeze_2263: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, 3);  unsqueeze_2262 = None
    mul_1225: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_107, unsqueeze_2263);  where_107 = unsqueeze_2263 = None
    mul_1226: "f32[256]" = torch.ops.aten.mul.Tensor(sum_217, rsqrt_107);  sum_217 = rsqrt_107 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    slice_570: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 0, 64)
    slice_571: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 64, 96)
    slice_572: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 96, 128)
    slice_573: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 128, 160)
    slice_574: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 160, 192)
    slice_575: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 192, 224)
    slice_576: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 224, 256);  mul_1225 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(slice_576, relu_12, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_576 = primals_39 = None
    getitem_323: "f32[4, 128, 56, 56]" = convolution_backward_107[0]
    getitem_324: "f32[32, 128, 3, 3]" = convolution_backward_107[1];  convolution_backward_107 = None
    le_108: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_108: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_108, full_default, getitem_323);  le_108 = getitem_323 = None
    add_864: "f32[128]" = torch.ops.aten.add.Tensor(primals_402, 1e-05);  primals_402 = None
    rsqrt_108: "f32[128]" = torch.ops.aten.rsqrt.default(add_864);  add_864 = None
    unsqueeze_2264: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_401, 0);  primals_401 = None
    unsqueeze_2265: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, 2);  unsqueeze_2264 = None
    unsqueeze_2266: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2265, 3);  unsqueeze_2265 = None
    sum_218: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_229: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_2266);  convolution_11 = unsqueeze_2266 = None
    mul_1227: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_108, sub_229);  sub_229 = None
    sum_219: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1227, [0, 2, 3]);  mul_1227 = None
    mul_1232: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_108, primals_37);  primals_37 = None
    unsqueeze_2273: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_2274: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2273, 2);  unsqueeze_2273 = None
    unsqueeze_2275: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, 3);  unsqueeze_2274 = None
    mul_1233: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_108, unsqueeze_2275);  where_108 = unsqueeze_2275 = None
    mul_1234: "f32[128]" = torch.ops.aten.mul.Tensor(sum_219, rsqrt_108);  sum_219 = rsqrt_108 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1233, relu_11, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1233 = primals_36 = None
    getitem_326: "f32[4, 224, 56, 56]" = convolution_backward_108[0]
    getitem_327: "f32[128, 224, 1, 1]" = convolution_backward_108[1];  convolution_backward_108 = None
    le_109: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_109: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_109, full_default, getitem_326);  le_109 = getitem_326 = None
    add_865: "f32[224]" = torch.ops.aten.add.Tensor(primals_399, 1e-05);  primals_399 = None
    rsqrt_109: "f32[224]" = torch.ops.aten.rsqrt.default(add_865);  add_865 = None
    unsqueeze_2276: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_398, 0);  primals_398 = None
    unsqueeze_2277: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, 2);  unsqueeze_2276 = None
    unsqueeze_2278: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2277, 3);  unsqueeze_2277 = None
    sum_220: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_230: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_2278);  cat_4 = unsqueeze_2278 = None
    mul_1235: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_109, sub_230);  sub_230 = None
    sum_221: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1235, [0, 2, 3]);  mul_1235 = None
    mul_1240: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_109, primals_34);  primals_34 = None
    unsqueeze_2285: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_2286: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2285, 2);  unsqueeze_2285 = None
    unsqueeze_2287: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, 3);  unsqueeze_2286 = None
    mul_1241: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_109, unsqueeze_2287);  where_109 = unsqueeze_2287 = None
    mul_1242: "f32[224]" = torch.ops.aten.mul.Tensor(sum_221, rsqrt_109);  sum_221 = rsqrt_109 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_577: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 0, 64)
    slice_578: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 64, 96)
    slice_579: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 96, 128)
    slice_580: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 128, 160)
    slice_581: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 160, 192)
    slice_582: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 192, 224);  mul_1241 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_866: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_570, slice_577);  slice_570 = slice_577 = None
    add_867: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_571, slice_578);  slice_571 = slice_578 = None
    add_868: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_572, slice_579);  slice_572 = slice_579 = None
    add_869: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_573, slice_580);  slice_573 = slice_580 = None
    add_870: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_574, slice_581);  slice_574 = slice_581 = None
    add_871: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_575, slice_582);  slice_575 = slice_582 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(add_871, relu_10, primals_33, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_871 = primals_33 = None
    getitem_329: "f32[4, 128, 56, 56]" = convolution_backward_109[0]
    getitem_330: "f32[32, 128, 3, 3]" = convolution_backward_109[1];  convolution_backward_109 = None
    le_110: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_110: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_110, full_default, getitem_329);  le_110 = getitem_329 = None
    add_872: "f32[128]" = torch.ops.aten.add.Tensor(primals_396, 1e-05);  primals_396 = None
    rsqrt_110: "f32[128]" = torch.ops.aten.rsqrt.default(add_872);  add_872 = None
    unsqueeze_2288: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_395, 0);  primals_395 = None
    unsqueeze_2289: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, 2);  unsqueeze_2288 = None
    unsqueeze_2290: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2289, 3);  unsqueeze_2289 = None
    sum_222: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_231: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_2290);  convolution_9 = unsqueeze_2290 = None
    mul_1243: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_110, sub_231);  sub_231 = None
    sum_223: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 2, 3]);  mul_1243 = None
    mul_1248: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_110, primals_31);  primals_31 = None
    unsqueeze_2297: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
    unsqueeze_2298: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2297, 2);  unsqueeze_2297 = None
    unsqueeze_2299: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, 3);  unsqueeze_2298 = None
    mul_1249: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_110, unsqueeze_2299);  where_110 = unsqueeze_2299 = None
    mul_1250: "f32[128]" = torch.ops.aten.mul.Tensor(sum_223, rsqrt_110);  sum_223 = rsqrt_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1249, relu_9, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1249 = primals_30 = None
    getitem_332: "f32[4, 192, 56, 56]" = convolution_backward_110[0]
    getitem_333: "f32[128, 192, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    le_111: "b8[4, 192, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_111: "f32[4, 192, 56, 56]" = torch.ops.aten.where.self(le_111, full_default, getitem_332);  le_111 = getitem_332 = None
    add_873: "f32[192]" = torch.ops.aten.add.Tensor(primals_393, 1e-05);  primals_393 = None
    rsqrt_111: "f32[192]" = torch.ops.aten.rsqrt.default(add_873);  add_873 = None
    unsqueeze_2300: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_392, 0);  primals_392 = None
    unsqueeze_2301: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2300, 2);  unsqueeze_2300 = None
    unsqueeze_2302: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2301, 3);  unsqueeze_2301 = None
    sum_224: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_232: "f32[4, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_2302);  cat_3 = unsqueeze_2302 = None
    mul_1251: "f32[4, 192, 56, 56]" = torch.ops.aten.mul.Tensor(where_111, sub_232);  sub_232 = None
    sum_225: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1251, [0, 2, 3]);  mul_1251 = None
    mul_1256: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_111, primals_28);  primals_28 = None
    unsqueeze_2309: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_2310: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2309, 2);  unsqueeze_2309 = None
    unsqueeze_2311: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, 3);  unsqueeze_2310 = None
    mul_1257: "f32[4, 192, 56, 56]" = torch.ops.aten.mul.Tensor(where_111, unsqueeze_2311);  where_111 = unsqueeze_2311 = None
    mul_1258: "f32[192]" = torch.ops.aten.mul.Tensor(sum_225, rsqrt_111);  sum_225 = rsqrt_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_583: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1257, 1, 0, 64)
    slice_584: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1257, 1, 64, 96)
    slice_585: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1257, 1, 96, 128)
    slice_586: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1257, 1, 128, 160)
    slice_587: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1257, 1, 160, 192);  mul_1257 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_874: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_866, slice_583);  add_866 = slice_583 = None
    add_875: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_867, slice_584);  add_867 = slice_584 = None
    add_876: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_868, slice_585);  add_868 = slice_585 = None
    add_877: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_869, slice_586);  add_869 = slice_586 = None
    add_878: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_870, slice_587);  add_870 = slice_587 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(add_878, relu_8, primals_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_878 = primals_27 = None
    getitem_335: "f32[4, 128, 56, 56]" = convolution_backward_111[0]
    getitem_336: "f32[32, 128, 3, 3]" = convolution_backward_111[1];  convolution_backward_111 = None
    le_112: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_112: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_112, full_default, getitem_335);  le_112 = getitem_335 = None
    add_879: "f32[128]" = torch.ops.aten.add.Tensor(primals_390, 1e-05);  primals_390 = None
    rsqrt_112: "f32[128]" = torch.ops.aten.rsqrt.default(add_879);  add_879 = None
    unsqueeze_2312: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_389, 0);  primals_389 = None
    unsqueeze_2313: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2312, 2);  unsqueeze_2312 = None
    unsqueeze_2314: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2313, 3);  unsqueeze_2313 = None
    sum_226: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_233: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_2314);  convolution_7 = unsqueeze_2314 = None
    mul_1259: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_112, sub_233);  sub_233 = None
    sum_227: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1259, [0, 2, 3]);  mul_1259 = None
    mul_1264: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_112, primals_25);  primals_25 = None
    unsqueeze_2321: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_2322: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2321, 2);  unsqueeze_2321 = None
    unsqueeze_2323: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, 3);  unsqueeze_2322 = None
    mul_1265: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_112, unsqueeze_2323);  where_112 = unsqueeze_2323 = None
    mul_1266: "f32[128]" = torch.ops.aten.mul.Tensor(sum_227, rsqrt_112);  sum_227 = rsqrt_112 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_1265, relu_7, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1265 = primals_24 = None
    getitem_338: "f32[4, 160, 56, 56]" = convolution_backward_112[0]
    getitem_339: "f32[128, 160, 1, 1]" = convolution_backward_112[1];  convolution_backward_112 = None
    le_113: "b8[4, 160, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_113: "f32[4, 160, 56, 56]" = torch.ops.aten.where.self(le_113, full_default, getitem_338);  le_113 = getitem_338 = None
    add_880: "f32[160]" = torch.ops.aten.add.Tensor(primals_387, 1e-05);  primals_387 = None
    rsqrt_113: "f32[160]" = torch.ops.aten.rsqrt.default(add_880);  add_880 = None
    unsqueeze_2324: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_386, 0);  primals_386 = None
    unsqueeze_2325: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2324, 2);  unsqueeze_2324 = None
    unsqueeze_2326: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2325, 3);  unsqueeze_2325 = None
    sum_228: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_234: "f32[4, 160, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_2326);  cat_2 = unsqueeze_2326 = None
    mul_1267: "f32[4, 160, 56, 56]" = torch.ops.aten.mul.Tensor(where_113, sub_234);  sub_234 = None
    sum_229: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1267, [0, 2, 3]);  mul_1267 = None
    mul_1272: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_113, primals_22);  primals_22 = None
    unsqueeze_2333: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_2334: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2333, 2);  unsqueeze_2333 = None
    unsqueeze_2335: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, 3);  unsqueeze_2334 = None
    mul_1273: "f32[4, 160, 56, 56]" = torch.ops.aten.mul.Tensor(where_113, unsqueeze_2335);  where_113 = unsqueeze_2335 = None
    mul_1274: "f32[160]" = torch.ops.aten.mul.Tensor(sum_229, rsqrt_113);  sum_229 = rsqrt_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_588: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1273, 1, 0, 64)
    slice_589: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1273, 1, 64, 96)
    slice_590: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1273, 1, 96, 128)
    slice_591: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1273, 1, 128, 160);  mul_1273 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_881: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_874, slice_588);  add_874 = slice_588 = None
    add_882: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_875, slice_589);  add_875 = slice_589 = None
    add_883: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_876, slice_590);  add_876 = slice_590 = None
    add_884: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_877, slice_591);  add_877 = slice_591 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(add_884, relu_6, primals_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_884 = primals_21 = None
    getitem_341: "f32[4, 128, 56, 56]" = convolution_backward_113[0]
    getitem_342: "f32[32, 128, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    le_114: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_114: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_114, full_default, getitem_341);  le_114 = getitem_341 = None
    add_885: "f32[128]" = torch.ops.aten.add.Tensor(primals_384, 1e-05);  primals_384 = None
    rsqrt_114: "f32[128]" = torch.ops.aten.rsqrt.default(add_885);  add_885 = None
    unsqueeze_2336: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_383, 0);  primals_383 = None
    unsqueeze_2337: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2336, 2);  unsqueeze_2336 = None
    unsqueeze_2338: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2337, 3);  unsqueeze_2337 = None
    sum_230: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_235: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_2338);  convolution_5 = unsqueeze_2338 = None
    mul_1275: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_114, sub_235);  sub_235 = None
    sum_231: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1275, [0, 2, 3]);  mul_1275 = None
    mul_1280: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_114, primals_19);  primals_19 = None
    unsqueeze_2345: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
    unsqueeze_2346: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2345, 2);  unsqueeze_2345 = None
    unsqueeze_2347: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, 3);  unsqueeze_2346 = None
    mul_1281: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_114, unsqueeze_2347);  where_114 = unsqueeze_2347 = None
    mul_1282: "f32[128]" = torch.ops.aten.mul.Tensor(sum_231, rsqrt_114);  sum_231 = rsqrt_114 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_1281, relu_5, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1281 = primals_18 = None
    getitem_344: "f32[4, 128, 56, 56]" = convolution_backward_114[0]
    getitem_345: "f32[128, 128, 1, 1]" = convolution_backward_114[1];  convolution_backward_114 = None
    le_115: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_115: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_115, full_default, getitem_344);  le_115 = getitem_344 = None
    add_886: "f32[128]" = torch.ops.aten.add.Tensor(primals_381, 1e-05);  primals_381 = None
    rsqrt_115: "f32[128]" = torch.ops.aten.rsqrt.default(add_886);  add_886 = None
    unsqueeze_2348: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_380, 0);  primals_380 = None
    unsqueeze_2349: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2348, 2);  unsqueeze_2348 = None
    unsqueeze_2350: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2349, 3);  unsqueeze_2349 = None
    sum_232: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_236: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_2350);  cat_1 = unsqueeze_2350 = None
    mul_1283: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_115, sub_236);  sub_236 = None
    sum_233: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1283, [0, 2, 3]);  mul_1283 = None
    mul_1288: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_115, primals_16);  primals_16 = None
    unsqueeze_2357: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1288, 0);  mul_1288 = None
    unsqueeze_2358: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2357, 2);  unsqueeze_2357 = None
    unsqueeze_2359: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, 3);  unsqueeze_2358 = None
    mul_1289: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_115, unsqueeze_2359);  where_115 = unsqueeze_2359 = None
    mul_1290: "f32[128]" = torch.ops.aten.mul.Tensor(sum_233, rsqrt_115);  sum_233 = rsqrt_115 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_592: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1289, 1, 0, 64)
    slice_593: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1289, 1, 64, 96)
    slice_594: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1289, 1, 96, 128);  mul_1289 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_887: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_881, slice_592);  add_881 = slice_592 = None
    add_888: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_882, slice_593);  add_882 = slice_593 = None
    add_889: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_883, slice_594);  add_883 = slice_594 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(add_889, relu_4, primals_15, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_889 = primals_15 = None
    getitem_347: "f32[4, 128, 56, 56]" = convolution_backward_115[0]
    getitem_348: "f32[32, 128, 3, 3]" = convolution_backward_115[1];  convolution_backward_115 = None
    le_116: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_116: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_116, full_default, getitem_347);  le_116 = getitem_347 = None
    add_890: "f32[128]" = torch.ops.aten.add.Tensor(primals_378, 1e-05);  primals_378 = None
    rsqrt_116: "f32[128]" = torch.ops.aten.rsqrt.default(add_890);  add_890 = None
    unsqueeze_2360: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_377, 0);  primals_377 = None
    unsqueeze_2361: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2360, 2);  unsqueeze_2360 = None
    unsqueeze_2362: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2361, 3);  unsqueeze_2361 = None
    sum_234: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_237: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_2362);  convolution_3 = unsqueeze_2362 = None
    mul_1291: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_116, sub_237);  sub_237 = None
    sum_235: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1291, [0, 2, 3]);  mul_1291 = None
    mul_1296: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_116, primals_13);  primals_13 = None
    unsqueeze_2369: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_2370: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2369, 2);  unsqueeze_2369 = None
    unsqueeze_2371: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, 3);  unsqueeze_2370 = None
    mul_1297: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_116, unsqueeze_2371);  where_116 = unsqueeze_2371 = None
    mul_1298: "f32[128]" = torch.ops.aten.mul.Tensor(sum_235, rsqrt_116);  sum_235 = rsqrt_116 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_1297, relu_3, primals_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1297 = primals_12 = None
    getitem_350: "f32[4, 96, 56, 56]" = convolution_backward_116[0]
    getitem_351: "f32[128, 96, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    le_117: "b8[4, 96, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_117: "f32[4, 96, 56, 56]" = torch.ops.aten.where.self(le_117, full_default, getitem_350);  le_117 = getitem_350 = None
    add_891: "f32[96]" = torch.ops.aten.add.Tensor(primals_375, 1e-05);  primals_375 = None
    rsqrt_117: "f32[96]" = torch.ops.aten.rsqrt.default(add_891);  add_891 = None
    unsqueeze_2372: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(primals_374, 0);  primals_374 = None
    unsqueeze_2373: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2372, 2);  unsqueeze_2372 = None
    unsqueeze_2374: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2373, 3);  unsqueeze_2373 = None
    sum_236: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_238: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_2374);  cat = unsqueeze_2374 = None
    mul_1299: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_117, sub_238);  sub_238 = None
    sum_237: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1299, [0, 2, 3]);  mul_1299 = None
    mul_1304: "f32[96]" = torch.ops.aten.mul.Tensor(rsqrt_117, primals_10);  primals_10 = None
    unsqueeze_2381: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
    unsqueeze_2382: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2381, 2);  unsqueeze_2381 = None
    unsqueeze_2383: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, 3);  unsqueeze_2382 = None
    mul_1305: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_117, unsqueeze_2383);  where_117 = unsqueeze_2383 = None
    mul_1306: "f32[96]" = torch.ops.aten.mul.Tensor(sum_237, rsqrt_117);  sum_237 = rsqrt_117 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_595: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1305, 1, 0, 64)
    slice_596: "f32[4, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1305, 1, 64, 96);  mul_1305 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_892: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_887, slice_595);  add_887 = slice_595 = None
    add_893: "f32[4, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_888, slice_596);  add_888 = slice_596 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(add_893, relu_2, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_893 = primals_9 = None
    getitem_353: "f32[4, 128, 56, 56]" = convolution_backward_117[0]
    getitem_354: "f32[32, 128, 3, 3]" = convolution_backward_117[1];  convolution_backward_117 = None
    le_118: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_118: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_118, full_default, getitem_353);  le_118 = getitem_353 = None
    add_894: "f32[128]" = torch.ops.aten.add.Tensor(primals_372, 1e-05);  primals_372 = None
    rsqrt_118: "f32[128]" = torch.ops.aten.rsqrt.default(add_894);  add_894 = None
    unsqueeze_2384: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_371, 0);  primals_371 = None
    unsqueeze_2385: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2384, 2);  unsqueeze_2384 = None
    unsqueeze_2386: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2385, 3);  unsqueeze_2385 = None
    sum_238: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_239: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_2386);  convolution_1 = unsqueeze_2386 = None
    mul_1307: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_118, sub_239);  sub_239 = None
    sum_239: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1307, [0, 2, 3]);  mul_1307 = None
    mul_1312: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_118, primals_7);  primals_7 = None
    unsqueeze_2393: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1312, 0);  mul_1312 = None
    unsqueeze_2394: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2393, 2);  unsqueeze_2393 = None
    unsqueeze_2395: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2394, 3);  unsqueeze_2394 = None
    mul_1313: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_118, unsqueeze_2395);  where_118 = unsqueeze_2395 = None
    mul_1314: "f32[128]" = torch.ops.aten.mul.Tensor(sum_239, rsqrt_118);  sum_239 = rsqrt_118 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1313, relu_1, primals_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1313 = primals_6 = None
    getitem_356: "f32[4, 64, 56, 56]" = convolution_backward_118[0]
    getitem_357: "f32[128, 64, 1, 1]" = convolution_backward_118[1];  convolution_backward_118 = None
    le_119: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_119: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_119, full_default, getitem_356);  le_119 = getitem_356 = None
    add_895: "f32[64]" = torch.ops.aten.add.Tensor(primals_369, 1e-05);  primals_369 = None
    rsqrt_119: "f32[64]" = torch.ops.aten.rsqrt.default(add_895);  add_895 = None
    sum_240: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    mul_1315: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_119, sub_240);  sub_240 = None
    sum_241: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 2, 3]);  mul_1315 = None
    mul_1320: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_119, primals_4);  primals_4 = None
    unsqueeze_2405: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
    unsqueeze_2406: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2405, 2);  unsqueeze_2405 = None
    unsqueeze_2407: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2406, 3);  unsqueeze_2406 = None
    mul_1321: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_119, unsqueeze_2407);  where_119 = unsqueeze_2407 = None
    mul_1322: "f32[64]" = torch.ops.aten.mul.Tensor(sum_241, rsqrt_119);  sum_241 = rsqrt_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    slice_597: "f32[4, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1321, 1, 0, 64);  mul_1321 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    add_896: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_892, slice_597);  add_892 = slice_597 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_896, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_896 = getitem_1 = None
    le_120: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_120: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_120, full_default, max_pool2d_with_indices_backward);  le_120 = full_default = max_pool2d_with_indices_backward = None
    add_897: "f32[64]" = torch.ops.aten.add.Tensor(primals_366, 1e-05);  primals_366 = None
    rsqrt_120: "f32[64]" = torch.ops.aten.rsqrt.default(add_897);  add_897 = None
    unsqueeze_2408: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_365, 0);  primals_365 = None
    unsqueeze_2409: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2408, 2);  unsqueeze_2408 = None
    unsqueeze_2410: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2409, 3);  unsqueeze_2409 = None
    sum_242: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_241: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_2410);  convolution = unsqueeze_2410 = None
    mul_1323: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_120, sub_241);  sub_241 = None
    sum_243: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1323, [0, 2, 3]);  mul_1323 = None
    mul_1328: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_120, primals_2);  primals_2 = None
    unsqueeze_2417: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
    unsqueeze_2418: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2417, 2);  unsqueeze_2417 = None
    unsqueeze_2419: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2418, 3);  unsqueeze_2418 = None
    mul_1329: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_120, unsqueeze_2419);  where_120 = unsqueeze_2419 = None
    mul_1330: "f32[64]" = torch.ops.aten.mul.Tensor(sum_243, rsqrt_120);  sum_243 = rsqrt_120 = None
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1329, primals_728, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1329 = primals_728 = primals_1 = None
    getitem_360: "f32[64, 3, 7, 7]" = convolution_backward_119[1];  convolution_backward_119 = None
    return [getitem_360, mul_1330, sum_242, mul_1322, sum_240, getitem_357, mul_1314, sum_238, getitem_354, mul_1306, sum_236, getitem_351, mul_1298, sum_234, getitem_348, mul_1290, sum_232, getitem_345, mul_1282, sum_230, getitem_342, mul_1274, sum_228, getitem_339, mul_1266, sum_226, getitem_336, mul_1258, sum_224, getitem_333, mul_1250, sum_222, getitem_330, mul_1242, sum_220, getitem_327, mul_1234, sum_218, getitem_324, mul_1226, sum_216, getitem_321, mul_1218, sum_214, getitem_318, mul_1210, sum_212, getitem_315, mul_1202, sum_210, getitem_312, mul_1194, sum_208, getitem_309, mul_1186, sum_206, getitem_306, mul_1178, sum_204, getitem_303, mul_1170, sum_202, getitem_300, mul_1162, sum_200, getitem_297, mul_1154, sum_198, getitem_294, mul_1146, sum_196, getitem_291, mul_1138, sum_194, getitem_288, mul_1130, sum_192, getitem_285, mul_1122, sum_190, getitem_282, mul_1114, sum_188, getitem_279, mul_1106, sum_186, getitem_276, mul_1098, sum_184, getitem_273, mul_1090, sum_182, getitem_270, mul_1082, sum_180, getitem_267, mul_1074, sum_178, getitem_264, mul_1066, sum_176, getitem_261, mul_1058, sum_174, getitem_258, mul_1050, sum_172, getitem_255, mul_1042, sum_170, getitem_252, mul_1034, sum_168, getitem_249, mul_1026, sum_166, getitem_246, mul_1018, sum_164, getitem_243, mul_1010, sum_162, getitem_240, mul_1002, sum_160, getitem_237, mul_994, sum_158, getitem_234, mul_986, sum_156, getitem_231, mul_978, sum_154, getitem_228, mul_970, sum_152, getitem_225, mul_962, sum_150, getitem_222, mul_954, sum_148, getitem_219, mul_946, sum_146, getitem_216, mul_938, sum_144, getitem_213, mul_930, sum_142, getitem_210, mul_922, sum_140, getitem_207, mul_914, sum_138, getitem_204, mul_906, sum_136, getitem_201, mul_898, sum_134, getitem_198, mul_890, sum_132, getitem_195, mul_882, sum_130, getitem_192, mul_874, sum_128, getitem_189, mul_866, sum_126, getitem_186, mul_858, sum_124, getitem_183, mul_850, sum_122, getitem_180, mul_842, sum_120, getitem_177, mul_834, sum_118, getitem_174, mul_826, sum_116, getitem_171, mul_818, sum_114, getitem_168, mul_810, sum_112, getitem_165, mul_802, sum_110, getitem_162, mul_794, sum_108, getitem_159, mul_786, sum_106, getitem_156, mul_778, sum_104, getitem_153, mul_770, sum_102, getitem_150, mul_762, sum_100, getitem_147, mul_754, sum_98, getitem_144, mul_746, sum_96, getitem_141, mul_738, sum_94, getitem_138, mul_730, sum_92, getitem_135, mul_722, sum_90, getitem_132, mul_714, sum_88, getitem_129, mul_706, sum_86, getitem_126, mul_698, sum_84, getitem_123, mul_690, sum_82, getitem_120, mul_682, sum_80, getitem_117, mul_674, sum_78, getitem_114, mul_666, sum_76, getitem_111, mul_658, sum_74, getitem_108, mul_650, sum_72, getitem_105, mul_642, sum_70, getitem_102, mul_634, sum_68, getitem_99, mul_626, sum_66, getitem_96, mul_618, sum_64, getitem_93, mul_610, sum_62, getitem_90, mul_602, sum_60, getitem_87, mul_594, sum_58, getitem_84, mul_586, sum_56, getitem_81, mul_578, sum_54, getitem_78, mul_570, sum_52, getitem_75, mul_562, sum_50, getitem_72, mul_554, sum_48, getitem_69, mul_546, sum_46, getitem_66, mul_538, sum_44, getitem_63, mul_530, sum_42, getitem_60, mul_522, sum_40, getitem_57, mul_514, sum_38, getitem_54, mul_506, sum_36, getitem_51, mul_498, sum_34, getitem_48, mul_490, sum_32, getitem_45, mul_482, sum_30, getitem_42, mul_474, sum_28, getitem_39, mul_466, sum_26, getitem_36, mul_458, sum_24, getitem_33, mul_450, sum_22, getitem_30, mul_442, sum_20, getitem_27, mul_434, sum_18, getitem_24, mul_426, sum_16, getitem_21, mul_418, sum_14, getitem_18, mul_410, sum_12, getitem_15, mul_402, sum_10, getitem_12, mul_394, sum_8, getitem_9, mul_386, sum_6, getitem_6, mul_378, sum_4, getitem_3, mul_370, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    