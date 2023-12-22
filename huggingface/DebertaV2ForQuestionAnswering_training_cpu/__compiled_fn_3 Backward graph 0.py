from __future__ import annotations



def forward(self, primals_3: "f32[1536]", primals_13: "f32[1536]", primals_19: "f32[1536]", primals_29: "f32[1536]", primals_35: "f32[1536]", primals_45: "f32[1536]", primals_51: "f32[1536]", primals_61: "f32[1536]", primals_67: "f32[1536]", primals_77: "f32[1536]", primals_83: "f32[1536]", primals_93: "f32[1536]", primals_99: "f32[1536]", primals_109: "f32[1536]", primals_115: "f32[1536]", primals_125: "f32[1536]", primals_131: "f32[1536]", primals_141: "f32[1536]", primals_147: "f32[1536]", primals_157: "f32[1536]", primals_163: "f32[1536]", primals_173: "f32[1536]", primals_179: "f32[1536]", primals_189: "f32[1536]", primals_195: "f32[1536]", primals_205: "f32[1536]", primals_211: "f32[1536]", primals_221: "f32[1536]", primals_227: "f32[1536]", primals_237: "f32[1536]", primals_243: "f32[1536]", primals_253: "f32[1536]", primals_259: "f32[1536]", primals_269: "f32[1536]", primals_275: "f32[1536]", primals_285: "f32[1536]", primals_291: "f32[1536]", primals_301: "f32[1536]", primals_307: "f32[1536]", primals_317: "f32[1536]", primals_323: "f32[1536]", primals_333: "f32[1536]", primals_339: "f32[1536]", primals_349: "f32[1536]", primals_355: "f32[1536]", primals_365: "f32[1536]", primals_371: "f32[1536]", primals_381: "f32[1536]", primals_387: "f32[1536]", primals_392: "i64[1, 512]", slice_1: "i64[1, 512]", mul: "f32[1, 512, 1536]", convert_element_type: "b8[1, 512, 1536]", view: "f32[512, 1536]", convert_element_type_2: "b8[1, 24, 512, 512]", view_16: "f32[512, 1536]", convert_element_type_3: "b8[1, 512, 1536]", mul_8: "f32[1, 512, 1536]", view_18: "f32[512, 1536]", addmm_4: "f32[512, 6144]", view_20: "f32[512, 6144]", convert_element_type_4: "b8[1, 512, 1536]", mul_14: "f32[1, 512, 1536]", view_22: "f32[512, 1536]", convert_element_type_6: "b8[1, 24, 512, 512]", view_38: "f32[512, 1536]", convert_element_type_7: "b8[1, 512, 1536]", mul_19: "f32[1, 512, 1536]", view_40: "f32[512, 1536]", addmm_10: "f32[512, 6144]", view_42: "f32[512, 6144]", convert_element_type_8: "b8[1, 512, 1536]", mul_25: "f32[1, 512, 1536]", view_44: "f32[512, 1536]", convert_element_type_10: "b8[1, 24, 512, 512]", view_60: "f32[512, 1536]", convert_element_type_11: "b8[1, 512, 1536]", mul_30: "f32[1, 512, 1536]", view_62: "f32[512, 1536]", addmm_16: "f32[512, 6144]", view_64: "f32[512, 6144]", convert_element_type_12: "b8[1, 512, 1536]", mul_36: "f32[1, 512, 1536]", view_66: "f32[512, 1536]", convert_element_type_14: "b8[1, 24, 512, 512]", view_82: "f32[512, 1536]", convert_element_type_15: "b8[1, 512, 1536]", mul_41: "f32[1, 512, 1536]", view_84: "f32[512, 1536]", addmm_22: "f32[512, 6144]", view_86: "f32[512, 6144]", convert_element_type_16: "b8[1, 512, 1536]", mul_47: "f32[1, 512, 1536]", view_88: "f32[512, 1536]", convert_element_type_18: "b8[1, 24, 512, 512]", view_104: "f32[512, 1536]", convert_element_type_19: "b8[1, 512, 1536]", mul_52: "f32[1, 512, 1536]", view_106: "f32[512, 1536]", addmm_28: "f32[512, 6144]", view_108: "f32[512, 6144]", convert_element_type_20: "b8[1, 512, 1536]", mul_58: "f32[1, 512, 1536]", view_110: "f32[512, 1536]", convert_element_type_22: "b8[1, 24, 512, 512]", view_126: "f32[512, 1536]", convert_element_type_23: "b8[1, 512, 1536]", mul_63: "f32[1, 512, 1536]", view_128: "f32[512, 1536]", addmm_34: "f32[512, 6144]", view_130: "f32[512, 6144]", convert_element_type_24: "b8[1, 512, 1536]", mul_69: "f32[1, 512, 1536]", view_132: "f32[512, 1536]", convert_element_type_26: "b8[1, 24, 512, 512]", view_148: "f32[512, 1536]", convert_element_type_27: "b8[1, 512, 1536]", mul_74: "f32[1, 512, 1536]", view_150: "f32[512, 1536]", addmm_40: "f32[512, 6144]", view_152: "f32[512, 6144]", convert_element_type_28: "b8[1, 512, 1536]", mul_80: "f32[1, 512, 1536]", view_154: "f32[512, 1536]", convert_element_type_30: "b8[1, 24, 512, 512]", view_170: "f32[512, 1536]", convert_element_type_31: "b8[1, 512, 1536]", mul_85: "f32[1, 512, 1536]", view_172: "f32[512, 1536]", addmm_46: "f32[512, 6144]", view_174: "f32[512, 6144]", convert_element_type_32: "b8[1, 512, 1536]", mul_91: "f32[1, 512, 1536]", view_176: "f32[512, 1536]", convert_element_type_34: "b8[1, 24, 512, 512]", view_192: "f32[512, 1536]", convert_element_type_35: "b8[1, 512, 1536]", mul_96: "f32[1, 512, 1536]", view_194: "f32[512, 1536]", addmm_52: "f32[512, 6144]", view_196: "f32[512, 6144]", convert_element_type_36: "b8[1, 512, 1536]", mul_102: "f32[1, 512, 1536]", view_198: "f32[512, 1536]", convert_element_type_38: "b8[1, 24, 512, 512]", view_214: "f32[512, 1536]", convert_element_type_39: "b8[1, 512, 1536]", mul_107: "f32[1, 512, 1536]", view_216: "f32[512, 1536]", addmm_58: "f32[512, 6144]", view_218: "f32[512, 6144]", convert_element_type_40: "b8[1, 512, 1536]", mul_113: "f32[1, 512, 1536]", view_220: "f32[512, 1536]", convert_element_type_42: "b8[1, 24, 512, 512]", view_236: "f32[512, 1536]", convert_element_type_43: "b8[1, 512, 1536]", mul_118: "f32[1, 512, 1536]", view_238: "f32[512, 1536]", addmm_64: "f32[512, 6144]", view_240: "f32[512, 6144]", convert_element_type_44: "b8[1, 512, 1536]", mul_124: "f32[1, 512, 1536]", view_242: "f32[512, 1536]", convert_element_type_46: "b8[1, 24, 512, 512]", view_258: "f32[512, 1536]", convert_element_type_47: "b8[1, 512, 1536]", mul_129: "f32[1, 512, 1536]", view_260: "f32[512, 1536]", addmm_70: "f32[512, 6144]", view_262: "f32[512, 6144]", convert_element_type_48: "b8[1, 512, 1536]", mul_135: "f32[1, 512, 1536]", view_264: "f32[512, 1536]", convert_element_type_50: "b8[1, 24, 512, 512]", view_280: "f32[512, 1536]", convert_element_type_51: "b8[1, 512, 1536]", mul_140: "f32[1, 512, 1536]", view_282: "f32[512, 1536]", addmm_76: "f32[512, 6144]", view_284: "f32[512, 6144]", convert_element_type_52: "b8[1, 512, 1536]", mul_146: "f32[1, 512, 1536]", view_286: "f32[512, 1536]", convert_element_type_54: "b8[1, 24, 512, 512]", view_302: "f32[512, 1536]", convert_element_type_55: "b8[1, 512, 1536]", mul_151: "f32[1, 512, 1536]", view_304: "f32[512, 1536]", addmm_82: "f32[512, 6144]", view_306: "f32[512, 6144]", convert_element_type_56: "b8[1, 512, 1536]", mul_157: "f32[1, 512, 1536]", view_308: "f32[512, 1536]", convert_element_type_58: "b8[1, 24, 512, 512]", view_324: "f32[512, 1536]", convert_element_type_59: "b8[1, 512, 1536]", mul_162: "f32[1, 512, 1536]", view_326: "f32[512, 1536]", addmm_88: "f32[512, 6144]", view_328: "f32[512, 6144]", convert_element_type_60: "b8[1, 512, 1536]", mul_168: "f32[1, 512, 1536]", view_330: "f32[512, 1536]", convert_element_type_62: "b8[1, 24, 512, 512]", view_346: "f32[512, 1536]", convert_element_type_63: "b8[1, 512, 1536]", mul_173: "f32[1, 512, 1536]", view_348: "f32[512, 1536]", addmm_94: "f32[512, 6144]", view_350: "f32[512, 6144]", convert_element_type_64: "b8[1, 512, 1536]", mul_179: "f32[1, 512, 1536]", view_352: "f32[512, 1536]", convert_element_type_66: "b8[1, 24, 512, 512]", view_368: "f32[512, 1536]", convert_element_type_67: "b8[1, 512, 1536]", mul_184: "f32[1, 512, 1536]", view_370: "f32[512, 1536]", addmm_100: "f32[512, 6144]", view_372: "f32[512, 6144]", convert_element_type_68: "b8[1, 512, 1536]", mul_190: "f32[1, 512, 1536]", view_374: "f32[512, 1536]", convert_element_type_70: "b8[1, 24, 512, 512]", view_390: "f32[512, 1536]", convert_element_type_71: "b8[1, 512, 1536]", mul_195: "f32[1, 512, 1536]", view_392: "f32[512, 1536]", addmm_106: "f32[512, 6144]", view_394: "f32[512, 6144]", convert_element_type_72: "b8[1, 512, 1536]", mul_201: "f32[1, 512, 1536]", view_396: "f32[512, 1536]", convert_element_type_74: "b8[1, 24, 512, 512]", view_412: "f32[512, 1536]", convert_element_type_75: "b8[1, 512, 1536]", mul_206: "f32[1, 512, 1536]", view_414: "f32[512, 1536]", addmm_112: "f32[512, 6144]", view_416: "f32[512, 6144]", convert_element_type_76: "b8[1, 512, 1536]", mul_212: "f32[1, 512, 1536]", view_418: "f32[512, 1536]", convert_element_type_78: "b8[1, 24, 512, 512]", view_434: "f32[512, 1536]", convert_element_type_79: "b8[1, 512, 1536]", mul_217: "f32[1, 512, 1536]", view_436: "f32[512, 1536]", addmm_118: "f32[512, 6144]", view_438: "f32[512, 6144]", convert_element_type_80: "b8[1, 512, 1536]", mul_223: "f32[1, 512, 1536]", view_440: "f32[512, 1536]", convert_element_type_82: "b8[1, 24, 512, 512]", view_456: "f32[512, 1536]", convert_element_type_83: "b8[1, 512, 1536]", mul_228: "f32[1, 512, 1536]", view_458: "f32[512, 1536]", addmm_124: "f32[512, 6144]", view_460: "f32[512, 6144]", convert_element_type_84: "b8[1, 512, 1536]", mul_234: "f32[1, 512, 1536]", view_462: "f32[512, 1536]", convert_element_type_86: "b8[1, 24, 512, 512]", view_478: "f32[512, 1536]", convert_element_type_87: "b8[1, 512, 1536]", mul_239: "f32[1, 512, 1536]", view_480: "f32[512, 1536]", addmm_130: "f32[512, 6144]", view_482: "f32[512, 6144]", convert_element_type_88: "b8[1, 512, 1536]", mul_245: "f32[1, 512, 1536]", view_484: "f32[512, 1536]", convert_element_type_90: "b8[1, 24, 512, 512]", view_500: "f32[512, 1536]", convert_element_type_91: "b8[1, 512, 1536]", mul_250: "f32[1, 512, 1536]", view_502: "f32[512, 1536]", addmm_136: "f32[512, 6144]", view_504: "f32[512, 6144]", convert_element_type_92: "b8[1, 512, 1536]", mul_256: "f32[1, 512, 1536]", view_506: "f32[512, 1536]", convert_element_type_94: "b8[1, 24, 512, 512]", view_522: "f32[512, 1536]", convert_element_type_95: "b8[1, 512, 1536]", mul_261: "f32[1, 512, 1536]", view_524: "f32[512, 1536]", addmm_142: "f32[512, 6144]", view_526: "f32[512, 6144]", convert_element_type_96: "b8[1, 512, 1536]", mul_267: "f32[1, 512, 1536]", view_528: "f32[512, 1536]", sub_147: "f32[1, 512]", ne: "b8[1]", sub_149: "f32[1, 512]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_125: "i64[1, 1]", ne_8: "b8[1, 1]", where_127: "i64[1, 1]", permute_338: "f32[2, 1536]", div_54: "f32[1, 512, 1]", permute_342: "f32[1536, 6144]", permute_346: "f32[6144, 1536]", div_55: "f32[1, 512, 1]", permute_350: "f32[1536, 1536]", permute_355: "f32[24, 512, 512]", permute_356: "f32[24, 64, 512]", alias_30: "f32[1, 24, 512, 512]", permute_357: "f32[24, 64, 512]", permute_358: "f32[24, 512, 64]", permute_361: "f32[1536, 1536]", permute_366: "f32[1536, 1536]", permute_371: "f32[1536, 1536]", div_57: "f32[1, 512, 1]", permute_375: "f32[1536, 6144]", permute_379: "f32[6144, 1536]", div_58: "f32[1, 512, 1]", permute_383: "f32[1536, 1536]", permute_388: "f32[24, 512, 512]", permute_389: "f32[24, 64, 512]", alias_33: "f32[1, 24, 512, 512]", permute_390: "f32[24, 64, 512]", permute_391: "f32[24, 512, 64]", permute_394: "f32[1536, 1536]", permute_399: "f32[1536, 1536]", permute_404: "f32[1536, 1536]", div_60: "f32[1, 512, 1]", permute_408: "f32[1536, 6144]", permute_412: "f32[6144, 1536]", div_61: "f32[1, 512, 1]", permute_416: "f32[1536, 1536]", permute_421: "f32[24, 512, 512]", permute_422: "f32[24, 64, 512]", alias_36: "f32[1, 24, 512, 512]", permute_423: "f32[24, 64, 512]", permute_424: "f32[24, 512, 64]", permute_427: "f32[1536, 1536]", permute_432: "f32[1536, 1536]", permute_437: "f32[1536, 1536]", div_63: "f32[1, 512, 1]", permute_441: "f32[1536, 6144]", permute_445: "f32[6144, 1536]", div_64: "f32[1, 512, 1]", permute_449: "f32[1536, 1536]", permute_454: "f32[24, 512, 512]", permute_455: "f32[24, 64, 512]", alias_39: "f32[1, 24, 512, 512]", permute_456: "f32[24, 64, 512]", permute_457: "f32[24, 512, 64]", permute_460: "f32[1536, 1536]", permute_465: "f32[1536, 1536]", permute_470: "f32[1536, 1536]", div_66: "f32[1, 512, 1]", permute_474: "f32[1536, 6144]", permute_478: "f32[6144, 1536]", div_67: "f32[1, 512, 1]", permute_482: "f32[1536, 1536]", permute_487: "f32[24, 512, 512]", permute_488: "f32[24, 64, 512]", alias_42: "f32[1, 24, 512, 512]", permute_489: "f32[24, 64, 512]", permute_490: "f32[24, 512, 64]", permute_493: "f32[1536, 1536]", permute_498: "f32[1536, 1536]", permute_503: "f32[1536, 1536]", div_69: "f32[1, 512, 1]", permute_507: "f32[1536, 6144]", permute_511: "f32[6144, 1536]", div_70: "f32[1, 512, 1]", permute_515: "f32[1536, 1536]", permute_520: "f32[24, 512, 512]", permute_521: "f32[24, 64, 512]", alias_45: "f32[1, 24, 512, 512]", permute_522: "f32[24, 64, 512]", permute_523: "f32[24, 512, 64]", permute_526: "f32[1536, 1536]", permute_531: "f32[1536, 1536]", permute_536: "f32[1536, 1536]", div_72: "f32[1, 512, 1]", permute_540: "f32[1536, 6144]", permute_544: "f32[6144, 1536]", div_73: "f32[1, 512, 1]", permute_548: "f32[1536, 1536]", permute_553: "f32[24, 512, 512]", permute_554: "f32[24, 64, 512]", alias_48: "f32[1, 24, 512, 512]", permute_555: "f32[24, 64, 512]", permute_556: "f32[24, 512, 64]", permute_559: "f32[1536, 1536]", permute_564: "f32[1536, 1536]", permute_569: "f32[1536, 1536]", div_75: "f32[1, 512, 1]", permute_573: "f32[1536, 6144]", permute_577: "f32[6144, 1536]", div_76: "f32[1, 512, 1]", permute_581: "f32[1536, 1536]", permute_586: "f32[24, 512, 512]", permute_587: "f32[24, 64, 512]", alias_51: "f32[1, 24, 512, 512]", permute_588: "f32[24, 64, 512]", permute_589: "f32[24, 512, 64]", permute_592: "f32[1536, 1536]", permute_597: "f32[1536, 1536]", permute_602: "f32[1536, 1536]", div_78: "f32[1, 512, 1]", permute_606: "f32[1536, 6144]", permute_610: "f32[6144, 1536]", div_79: "f32[1, 512, 1]", permute_614: "f32[1536, 1536]", permute_619: "f32[24, 512, 512]", permute_620: "f32[24, 64, 512]", alias_54: "f32[1, 24, 512, 512]", permute_621: "f32[24, 64, 512]", permute_622: "f32[24, 512, 64]", permute_625: "f32[1536, 1536]", permute_630: "f32[1536, 1536]", permute_635: "f32[1536, 1536]", div_81: "f32[1, 512, 1]", permute_639: "f32[1536, 6144]", permute_643: "f32[6144, 1536]", div_82: "f32[1, 512, 1]", permute_647: "f32[1536, 1536]", permute_652: "f32[24, 512, 512]", permute_653: "f32[24, 64, 512]", alias_57: "f32[1, 24, 512, 512]", permute_654: "f32[24, 64, 512]", permute_655: "f32[24, 512, 64]", permute_658: "f32[1536, 1536]", permute_663: "f32[1536, 1536]", permute_668: "f32[1536, 1536]", div_84: "f32[1, 512, 1]", permute_672: "f32[1536, 6144]", permute_676: "f32[6144, 1536]", div_85: "f32[1, 512, 1]", permute_680: "f32[1536, 1536]", permute_685: "f32[24, 512, 512]", permute_686: "f32[24, 64, 512]", alias_60: "f32[1, 24, 512, 512]", permute_687: "f32[24, 64, 512]", permute_688: "f32[24, 512, 64]", permute_691: "f32[1536, 1536]", permute_696: "f32[1536, 1536]", permute_701: "f32[1536, 1536]", div_87: "f32[1, 512, 1]", permute_705: "f32[1536, 6144]", permute_709: "f32[6144, 1536]", div_88: "f32[1, 512, 1]", permute_713: "f32[1536, 1536]", permute_718: "f32[24, 512, 512]", permute_719: "f32[24, 64, 512]", alias_63: "f32[1, 24, 512, 512]", permute_720: "f32[24, 64, 512]", permute_721: "f32[24, 512, 64]", permute_724: "f32[1536, 1536]", permute_729: "f32[1536, 1536]", permute_734: "f32[1536, 1536]", div_90: "f32[1, 512, 1]", permute_738: "f32[1536, 6144]", permute_742: "f32[6144, 1536]", div_91: "f32[1, 512, 1]", permute_746: "f32[1536, 1536]", permute_751: "f32[24, 512, 512]", permute_752: "f32[24, 64, 512]", alias_66: "f32[1, 24, 512, 512]", permute_753: "f32[24, 64, 512]", permute_754: "f32[24, 512, 64]", permute_757: "f32[1536, 1536]", permute_762: "f32[1536, 1536]", permute_767: "f32[1536, 1536]", div_93: "f32[1, 512, 1]", permute_771: "f32[1536, 6144]", permute_775: "f32[6144, 1536]", div_94: "f32[1, 512, 1]", permute_779: "f32[1536, 1536]", permute_784: "f32[24, 512, 512]", permute_785: "f32[24, 64, 512]", alias_69: "f32[1, 24, 512, 512]", permute_786: "f32[24, 64, 512]", permute_787: "f32[24, 512, 64]", permute_790: "f32[1536, 1536]", permute_795: "f32[1536, 1536]", permute_800: "f32[1536, 1536]", div_96: "f32[1, 512, 1]", permute_804: "f32[1536, 6144]", permute_808: "f32[6144, 1536]", div_97: "f32[1, 512, 1]", permute_812: "f32[1536, 1536]", permute_817: "f32[24, 512, 512]", permute_818: "f32[24, 64, 512]", alias_72: "f32[1, 24, 512, 512]", permute_819: "f32[24, 64, 512]", permute_820: "f32[24, 512, 64]", permute_823: "f32[1536, 1536]", permute_828: "f32[1536, 1536]", permute_833: "f32[1536, 1536]", div_99: "f32[1, 512, 1]", permute_837: "f32[1536, 6144]", permute_841: "f32[6144, 1536]", div_100: "f32[1, 512, 1]", permute_845: "f32[1536, 1536]", permute_850: "f32[24, 512, 512]", permute_851: "f32[24, 64, 512]", alias_75: "f32[1, 24, 512, 512]", permute_852: "f32[24, 64, 512]", permute_853: "f32[24, 512, 64]", permute_856: "f32[1536, 1536]", permute_861: "f32[1536, 1536]", permute_866: "f32[1536, 1536]", div_102: "f32[1, 512, 1]", permute_870: "f32[1536, 6144]", permute_874: "f32[6144, 1536]", div_103: "f32[1, 512, 1]", permute_878: "f32[1536, 1536]", permute_883: "f32[24, 512, 512]", permute_884: "f32[24, 64, 512]", alias_78: "f32[1, 24, 512, 512]", permute_885: "f32[24, 64, 512]", permute_886: "f32[24, 512, 64]", permute_889: "f32[1536, 1536]", permute_894: "f32[1536, 1536]", permute_899: "f32[1536, 1536]", div_105: "f32[1, 512, 1]", permute_903: "f32[1536, 6144]", permute_907: "f32[6144, 1536]", div_106: "f32[1, 512, 1]", permute_911: "f32[1536, 1536]", permute_916: "f32[24, 512, 512]", permute_917: "f32[24, 64, 512]", alias_81: "f32[1, 24, 512, 512]", permute_918: "f32[24, 64, 512]", permute_919: "f32[24, 512, 64]", permute_922: "f32[1536, 1536]", permute_927: "f32[1536, 1536]", permute_932: "f32[1536, 1536]", div_108: "f32[1, 512, 1]", permute_936: "f32[1536, 6144]", permute_940: "f32[6144, 1536]", div_109: "f32[1, 512, 1]", permute_944: "f32[1536, 1536]", permute_949: "f32[24, 512, 512]", permute_950: "f32[24, 64, 512]", alias_84: "f32[1, 24, 512, 512]", permute_951: "f32[24, 64, 512]", permute_952: "f32[24, 512, 64]", permute_955: "f32[1536, 1536]", permute_960: "f32[1536, 1536]", permute_965: "f32[1536, 1536]", div_111: "f32[1, 512, 1]", permute_969: "f32[1536, 6144]", permute_973: "f32[6144, 1536]", div_112: "f32[1, 512, 1]", permute_977: "f32[1536, 1536]", permute_982: "f32[24, 512, 512]", permute_983: "f32[24, 64, 512]", alias_87: "f32[1, 24, 512, 512]", permute_984: "f32[24, 64, 512]", permute_985: "f32[24, 512, 64]", permute_988: "f32[1536, 1536]", permute_993: "f32[1536, 1536]", permute_998: "f32[1536, 1536]", div_114: "f32[1, 512, 1]", permute_1002: "f32[1536, 6144]", permute_1006: "f32[6144, 1536]", div_115: "f32[1, 512, 1]", permute_1010: "f32[1536, 1536]", permute_1015: "f32[24, 512, 512]", permute_1016: "f32[24, 64, 512]", alias_90: "f32[1, 24, 512, 512]", permute_1017: "f32[24, 64, 512]", permute_1018: "f32[24, 512, 64]", permute_1021: "f32[1536, 1536]", permute_1026: "f32[1536, 1536]", permute_1031: "f32[1536, 1536]", div_117: "f32[1, 512, 1]", permute_1035: "f32[1536, 6144]", permute_1039: "f32[6144, 1536]", div_118: "f32[1, 512, 1]", permute_1043: "f32[1536, 1536]", permute_1048: "f32[24, 512, 512]", permute_1049: "f32[24, 64, 512]", alias_93: "f32[1, 24, 512, 512]", permute_1050: "f32[24, 64, 512]", permute_1051: "f32[24, 512, 64]", permute_1054: "f32[1536, 1536]", permute_1059: "f32[1536, 1536]", permute_1064: "f32[1536, 1536]", div_120: "f32[1, 512, 1]", permute_1068: "f32[1536, 6144]", permute_1072: "f32[6144, 1536]", div_121: "f32[1, 512, 1]", permute_1076: "f32[1536, 1536]", permute_1081: "f32[24, 512, 512]", permute_1082: "f32[24, 64, 512]", alias_96: "f32[1, 24, 512, 512]", permute_1083: "f32[24, 64, 512]", permute_1084: "f32[24, 512, 64]", permute_1087: "f32[1536, 1536]", permute_1092: "f32[1536, 1536]", permute_1097: "f32[1536, 1536]", div_123: "f32[1, 512, 1]", permute_1101: "f32[1536, 6144]", permute_1105: "f32[6144, 1536]", div_124: "f32[1, 512, 1]", permute_1109: "f32[1536, 1536]", permute_1114: "f32[24, 512, 512]", permute_1115: "f32[24, 64, 512]", alias_99: "f32[1, 24, 512, 512]", permute_1116: "f32[24, 64, 512]", permute_1117: "f32[24, 512, 64]", permute_1120: "f32[1536, 1536]", permute_1125: "f32[1536, 1536]", permute_1130: "f32[1536, 1536]", div_126: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512]", tangents_3: "f32[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:723, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_4, [1, 512, 6144]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_11: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_6: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_10, [1, 512, 6144]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
    add_13: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_16, [1, 512, 6144]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_20: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_22, [1, 512, 6144]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
    add_27: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_28, [1, 512, 6144]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_34: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_34, [1, 512, 6144]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_66: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_66);  mul_66 = None
    add_41: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_40, [1, 512, 6144]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_77: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_48: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_46, [1, 512, 6144]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_55: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_52, [1, 512, 6144]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_99: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_62: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_58, [1, 512, 6144]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_110: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_69: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_64, [1, 512, 6144]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_121: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_76: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_70, [1, 512, 6144]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_132: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_83: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_76, [1, 512, 6144]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_143: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
    erf_12: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_90: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_82, [1, 512, 6144]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_154: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
    erf_13: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_97: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_327: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_88, [1, 512, 6144]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_165: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_14: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_104: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_349: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_94, [1, 512, 6144]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_176: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
    erf_15: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_111: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_371: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_100, [1, 512, 6144]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_187: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_16: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_118: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_393: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_106, [1, 512, 6144]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_198: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_17: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
    add_125: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_415: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_112, [1, 512, 6144]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_209: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_18: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_132: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_118, [1, 512, 6144]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_220: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
    erf_19: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
    add_139: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_459: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_124, [1, 512, 6144]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_231: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
    erf_20: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_231);  mul_231 = None
    add_146: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_130, [1, 512, 6144]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_242: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
    erf_21: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_153: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_503: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_136, [1, 512, 6144]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_253: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
    erf_22: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_253);  mul_253 = None
    add_160: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_525: "f32[1, 512, 6144]" = torch.ops.aten.view.default(addmm_142, [1, 512, 6144]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_264: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
    erf_23: "f32[1, 512, 6144]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
    add_167: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    alias_24: "f32[1, 512]" = torch.ops.aten.alias.default(sub_147);  sub_147 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_97: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    alias_25: "f32[1, 512]" = torch.ops.aten.alias.default(sub_149);  sub_149 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_98: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1533, code: total_loss = (start_loss + end_loss) / 2
    div_51: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    div_52: "f32[]" = torch.ops.aten.div.Tensor(div_51, convert_element_type_98);  convert_element_type_98 = None
    full_default_175: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_175, 1, where_125, -1.0);  where_125 = None
    where_126: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_52, full_default_1);  ne_6 = div_52 = None
    mul_269: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_126);  scatter = where_126 = None
    alias_26: "f32[1, 512]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    exp_26: "f32[1, 512]" = torch.ops.aten.exp.default(alias_26);  alias_26 = None
    sum_31: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [1], True)
    mul_270: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_26, sum_31);  exp_26 = sum_31 = None
    sub_150: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1532, code: end_loss = loss_fct(end_logits, end_positions)
    add_172: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_150);  tangents_3 = sub_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    div_53: "f32[]" = torch.ops.aten.div.Tensor(div_51, convert_element_type_97);  div_51 = convert_element_type_97 = None
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_175, 1, where_127, -1.0);  full_default_175 = where_127 = None
    where_128: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_53, full_default_1);  ne_8 = div_53 = None
    mul_271: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_128);  scatter_1 = where_128 = None
    alias_27: "f32[1, 512]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_27: "f32[1, 512]" = torch.ops.aten.exp.default(alias_27);  alias_27 = None
    sum_32: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [1], True)
    mul_272: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_27, sum_32);  exp_27 = sum_32 = None
    sub_151: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1531, code: start_loss = loss_fct(start_logits, start_positions)
    add_173: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_151);  tangents_2 = sub_151 = None
    
    # No stacktrace found for following nodes
    unsqueeze_8: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_172, 2);  add_172 = None
    unsqueeze_9: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_173, 2);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1514, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_9, unsqueeze_8], 2);  unsqueeze_9 = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:1513, code: logits = self.qa_outputs(sequence_output)
    view_530: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    mm: "f32[512, 1536]" = torch.ops.aten.mm.default(view_530, permute_338);  permute_338 = None
    permute_339: "f32[2, 512]" = torch.ops.aten.permute.default(view_530, [1, 0])
    mm_1: "f32[2, 1536]" = torch.ops.aten.mm.default(permute_339, view_528);  permute_339 = view_528 = None
    permute_340: "f32[1536, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_33: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
    view_531: "f32[2]" = torch.ops.aten.view.default(sum_33, [2]);  sum_33 = None
    permute_341: "f32[2, 1536]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_532: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm, [1, 512, 1536]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_274: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(view_532, primals_387);  primals_387 = None
    mul_275: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_274, 1536)
    sum_34: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
    mul_276: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_274, mul_267);  mul_274 = None
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
    mul_277: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_267, sum_35);  sum_35 = None
    sub_153: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_275, sum_34);  mul_275 = sum_34 = None
    sub_154: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_153, mul_277);  sub_153 = mul_277 = None
    mul_278: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_54, sub_154);  div_54 = sub_154 = None
    mul_279: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(view_532, mul_267);  mul_267 = None
    sum_36: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
    sum_37: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_532, [0, 1]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_129: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_96, full_default_1, mul_278);  convert_element_type_96 = None
    mul_280: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_129, 1.1111111111111112);  where_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_533: "f32[512, 1536]" = torch.ops.aten.view.default(mul_280, [512, 1536]);  mul_280 = None
    mm_2: "f32[512, 6144]" = torch.ops.aten.mm.default(view_533, permute_342);  permute_342 = None
    permute_343: "f32[1536, 512]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_3: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_343, view_526);  permute_343 = view_526 = None
    permute_344: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_38: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
    view_534: "f32[1536]" = torch.ops.aten.view.default(sum_38, [1536]);  sum_38 = None
    permute_345: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    view_535: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_2, [1, 512, 6144]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_282: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_167, 0.5);  add_167 = None
    mul_283: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, view_525)
    mul_284: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_283, -0.5);  mul_283 = None
    exp_28: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_284);  mul_284 = None
    mul_285: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_286: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, mul_285);  view_525 = mul_285 = None
    add_175: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_282, mul_286);  mul_282 = mul_286 = None
    mul_287: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_535, add_175);  view_535 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_536: "f32[512, 6144]" = torch.ops.aten.view.default(mul_287, [512, 6144]);  mul_287 = None
    mm_4: "f32[512, 1536]" = torch.ops.aten.mm.default(view_536, permute_346);  permute_346 = None
    permute_347: "f32[6144, 512]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_5: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_347, view_524);  permute_347 = view_524 = None
    permute_348: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_39: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[6144]" = torch.ops.aten.view.default(sum_39, [6144]);  sum_39 = None
    permute_349: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_538: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_4, [1, 512, 1536]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_176: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_278, view_538);  mul_278 = view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_289: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_176, primals_381);  primals_381 = None
    mul_290: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_289, 1536)
    sum_40: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True)
    mul_291: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_289, mul_261);  mul_289 = None
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
    mul_292: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_261, sum_41);  sum_41 = None
    sub_156: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_290, sum_40);  mul_290 = sum_40 = None
    sub_157: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_156, mul_292);  sub_156 = mul_292 = None
    mul_293: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_55, sub_157);  div_55 = sub_157 = None
    mul_294: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_176, mul_261);  mul_261 = None
    sum_42: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1]);  mul_294 = None
    sum_43: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_130: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_95, full_default_1, mul_293);  convert_element_type_95 = None
    mul_295: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_130, 1.1111111111111112);  where_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_539: "f32[512, 1536]" = torch.ops.aten.view.default(mul_295, [512, 1536]);  mul_295 = None
    mm_6: "f32[512, 1536]" = torch.ops.aten.mm.default(view_539, permute_350);  permute_350 = None
    permute_351: "f32[1536, 512]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_7: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_351, view_522);  permute_351 = view_522 = None
    permute_352: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_44: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[1536]" = torch.ops.aten.view.default(sum_44, [1536]);  sum_44 = None
    permute_353: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_541: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_6, [1, 512, 1536]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_542: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_541, [1, 512, 24, 64]);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_354: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_543: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_354, [24, 512, 64]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_48: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_355, view_543);  permute_355 = None
    bmm_49: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_543, permute_356);  view_543 = permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_544: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 24, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_131: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_94, full_default_1, view_544);  convert_element_type_94 = view_544 = None
    mul_296: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_131, 1.1111111111111112);  where_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_297: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_296, alias_30);  mul_296 = None
    sum_45: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [-1], True)
    mul_298: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_30, sum_45);  alias_30 = sum_45 = None
    sub_158: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_545: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_158, [24, 512, 512]);  sub_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_50: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_357, view_545);  permute_357 = None
    bmm_51: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_545, permute_358);  view_545 = permute_358 = None
    div_56: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_50, full_default_2);  bmm_50 = None
    permute_359: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_56, [0, 2, 1]);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_546: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 24, 512, 64]);  bmm_48 = None
    permute_360: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_98: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_547: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_98, [1, 512, 1536]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_548: "f32[512, 1536]" = torch.ops.aten.view.default(view_547, [512, 1536]);  view_547 = None
    mm_8: "f32[512, 1536]" = torch.ops.aten.mm.default(view_548, permute_361);  permute_361 = None
    permute_362: "f32[1536, 512]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_9: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_362, view_506);  permute_362 = None
    permute_363: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_46: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[1536]" = torch.ops.aten.view.default(sum_46, [1536]);  sum_46 = None
    permute_364: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_550: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_8, [1, 512, 1536]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_177: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_293, view_550);  mul_293 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_551: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_359, [1, 24, 512, 64]);  permute_359 = None
    permute_365: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_552: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_365, [1, 512, 1536]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_553: "f32[512, 1536]" = torch.ops.aten.view.default(view_552, [512, 1536]);  view_552 = None
    mm_10: "f32[512, 1536]" = torch.ops.aten.mm.default(view_553, permute_366);  permute_366 = None
    permute_367: "f32[1536, 512]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_11: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_367, view_506);  permute_367 = None
    permute_368: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_47: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[1536]" = torch.ops.aten.view.default(sum_47, [1536]);  sum_47 = None
    permute_369: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_555: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_10, [1, 512, 1536]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_178: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_177, view_555);  add_177 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_556: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 24, 512, 64]);  bmm_51 = None
    permute_370: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_99: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_557: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_99, [1, 512, 1536]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_558: "f32[512, 1536]" = torch.ops.aten.view.default(view_557, [512, 1536]);  view_557 = None
    mm_12: "f32[512, 1536]" = torch.ops.aten.mm.default(view_558, permute_371);  permute_371 = None
    permute_372: "f32[1536, 512]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_13: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_372, view_506);  permute_372 = view_506 = None
    permute_373: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_48: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
    view_559: "f32[1536]" = torch.ops.aten.view.default(sum_48, [1536]);  sum_48 = None
    permute_374: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    view_560: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_12, [1, 512, 1536]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_179: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_178, view_560);  add_178 = view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_300: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_179, primals_371);  primals_371 = None
    mul_301: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_300, 1536)
    sum_49: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_300, mul_256);  mul_300 = None
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_256, sum_50);  sum_50 = None
    sub_160: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_301, sum_49);  mul_301 = sum_49 = None
    sub_161: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_160, mul_303);  sub_160 = mul_303 = None
    mul_304: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_57, sub_161);  div_57 = sub_161 = None
    mul_305: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_179, mul_256);  mul_256 = None
    sum_51: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_52: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_132: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_92, full_default_1, mul_304);  convert_element_type_92 = None
    mul_306: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_132, 1.1111111111111112);  where_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_561: "f32[512, 1536]" = torch.ops.aten.view.default(mul_306, [512, 1536]);  mul_306 = None
    mm_14: "f32[512, 6144]" = torch.ops.aten.mm.default(view_561, permute_375);  permute_375 = None
    permute_376: "f32[1536, 512]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_15: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_376, view_504);  permute_376 = view_504 = None
    permute_377: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_53: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[1536]" = torch.ops.aten.view.default(sum_53, [1536]);  sum_53 = None
    permute_378: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_563: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_14, [1, 512, 6144]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_308: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_309: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, view_503)
    mul_310: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_309, -0.5);  mul_309 = None
    exp_29: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_310);  mul_310 = None
    mul_311: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_312: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, mul_311);  view_503 = mul_311 = None
    add_181: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_308, mul_312);  mul_308 = mul_312 = None
    mul_313: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_563, add_181);  view_563 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_564: "f32[512, 6144]" = torch.ops.aten.view.default(mul_313, [512, 6144]);  mul_313 = None
    mm_16: "f32[512, 1536]" = torch.ops.aten.mm.default(view_564, permute_379);  permute_379 = None
    permute_380: "f32[6144, 512]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_17: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_380, view_502);  permute_380 = view_502 = None
    permute_381: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_54: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[6144]" = torch.ops.aten.view.default(sum_54, [6144]);  sum_54 = None
    permute_382: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_566: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_16, [1, 512, 1536]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_182: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_304, view_566);  mul_304 = view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_315: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_182, primals_365);  primals_365 = None
    mul_316: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_315, 1536)
    sum_55: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True)
    mul_317: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_315, mul_250);  mul_315 = None
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    mul_318: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_250, sum_56);  sum_56 = None
    sub_163: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_316, sum_55);  mul_316 = sum_55 = None
    sub_164: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_163, mul_318);  sub_163 = mul_318 = None
    mul_319: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_58, sub_164);  div_58 = sub_164 = None
    mul_320: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_182, mul_250);  mul_250 = None
    sum_57: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1]);  mul_320 = None
    sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_133: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_91, full_default_1, mul_319);  convert_element_type_91 = None
    mul_321: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_133, 1.1111111111111112);  where_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_567: "f32[512, 1536]" = torch.ops.aten.view.default(mul_321, [512, 1536]);  mul_321 = None
    mm_18: "f32[512, 1536]" = torch.ops.aten.mm.default(view_567, permute_383);  permute_383 = None
    permute_384: "f32[1536, 512]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_19: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_384, view_500);  permute_384 = view_500 = None
    permute_385: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_59: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[1536]" = torch.ops.aten.view.default(sum_59, [1536]);  sum_59 = None
    permute_386: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_569: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_18, [1, 512, 1536]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_570: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_569, [1, 512, 24, 64]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_387: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_571: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_387, [24, 512, 64]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_52: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_388, view_571);  permute_388 = None
    bmm_53: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_571, permute_389);  view_571 = permute_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_572: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 24, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_134: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_90, full_default_1, view_572);  convert_element_type_90 = view_572 = None
    mul_322: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_134, 1.1111111111111112);  where_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_323: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_322, alias_33);  mul_322 = None
    sum_60: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
    mul_324: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_60);  alias_33 = sum_60 = None
    sub_165: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_573: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_165, [24, 512, 512]);  sub_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_54: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_390, view_573);  permute_390 = None
    bmm_55: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_573, permute_391);  view_573 = permute_391 = None
    div_59: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_54, full_default_2);  bmm_54 = None
    permute_392: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_59, [0, 2, 1]);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_574: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 24, 512, 64]);  bmm_52 = None
    permute_393: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_100: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_575: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_100, [1, 512, 1536]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_576: "f32[512, 1536]" = torch.ops.aten.view.default(view_575, [512, 1536]);  view_575 = None
    mm_20: "f32[512, 1536]" = torch.ops.aten.mm.default(view_576, permute_394);  permute_394 = None
    permute_395: "f32[1536, 512]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_21: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_395, view_484);  permute_395 = None
    permute_396: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_61: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[1536]" = torch.ops.aten.view.default(sum_61, [1536]);  sum_61 = None
    permute_397: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_578: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_20, [1, 512, 1536]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_183: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_319, view_578);  mul_319 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_579: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_392, [1, 24, 512, 64]);  permute_392 = None
    permute_398: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_580: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_398, [1, 512, 1536]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_581: "f32[512, 1536]" = torch.ops.aten.view.default(view_580, [512, 1536]);  view_580 = None
    mm_22: "f32[512, 1536]" = torch.ops.aten.mm.default(view_581, permute_399);  permute_399 = None
    permute_400: "f32[1536, 512]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_23: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_400, view_484);  permute_400 = None
    permute_401: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[1536]" = torch.ops.aten.view.default(sum_62, [1536]);  sum_62 = None
    permute_402: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    view_583: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_22, [1, 512, 1536]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_184: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_183, view_583);  add_183 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_584: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 24, 512, 64]);  bmm_55 = None
    permute_403: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_101: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
    view_585: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_101, [1, 512, 1536]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_586: "f32[512, 1536]" = torch.ops.aten.view.default(view_585, [512, 1536]);  view_585 = None
    mm_24: "f32[512, 1536]" = torch.ops.aten.mm.default(view_586, permute_404);  permute_404 = None
    permute_405: "f32[1536, 512]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_25: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_405, view_484);  permute_405 = view_484 = None
    permute_406: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_63: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
    view_587: "f32[1536]" = torch.ops.aten.view.default(sum_63, [1536]);  sum_63 = None
    permute_407: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_588: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_24, [1, 512, 1536]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_185: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_184, view_588);  add_184 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_326: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_185, primals_355);  primals_355 = None
    mul_327: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_326, 1536)
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
    mul_328: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_326, mul_245);  mul_326 = None
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
    mul_329: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_245, sum_65);  sum_65 = None
    sub_167: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_327, sum_64);  mul_327 = sum_64 = None
    sub_168: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_167, mul_329);  sub_167 = mul_329 = None
    mul_330: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_60, sub_168);  div_60 = sub_168 = None
    mul_331: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_185, mul_245);  mul_245 = None
    sum_66: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
    sum_67: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_135: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_88, full_default_1, mul_330);  convert_element_type_88 = None
    mul_332: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_135, 1.1111111111111112);  where_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_589: "f32[512, 1536]" = torch.ops.aten.view.default(mul_332, [512, 1536]);  mul_332 = None
    mm_26: "f32[512, 6144]" = torch.ops.aten.mm.default(view_589, permute_408);  permute_408 = None
    permute_409: "f32[1536, 512]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_27: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_409, view_482);  permute_409 = view_482 = None
    permute_410: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_68: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[1536]" = torch.ops.aten.view.default(sum_68, [1536]);  sum_68 = None
    permute_411: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_591: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_26, [1, 512, 6144]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_334: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_153, 0.5);  add_153 = None
    mul_335: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, view_481)
    mul_336: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_335, -0.5);  mul_335 = None
    exp_30: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_336);  mul_336 = None
    mul_337: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_338: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, mul_337);  view_481 = mul_337 = None
    add_187: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_334, mul_338);  mul_334 = mul_338 = None
    mul_339: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_591, add_187);  view_591 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_592: "f32[512, 6144]" = torch.ops.aten.view.default(mul_339, [512, 6144]);  mul_339 = None
    mm_28: "f32[512, 1536]" = torch.ops.aten.mm.default(view_592, permute_412);  permute_412 = None
    permute_413: "f32[6144, 512]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_29: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_413, view_480);  permute_413 = view_480 = None
    permute_414: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_69: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[6144]" = torch.ops.aten.view.default(sum_69, [6144]);  sum_69 = None
    permute_415: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    view_594: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_28, [1, 512, 1536]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_188: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_330, view_594);  mul_330 = view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_341: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_188, primals_349);  primals_349 = None
    mul_342: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_341, 1536)
    sum_70: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_341, mul_239);  mul_341 = None
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_239, sum_71);  sum_71 = None
    sub_170: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_342, sum_70);  mul_342 = sum_70 = None
    sub_171: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_170, mul_344);  sub_170 = mul_344 = None
    mul_345: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_61, sub_171);  div_61 = sub_171 = None
    mul_346: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_188, mul_239);  mul_239 = None
    sum_72: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_73: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_188, [0, 1]);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_136: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_87, full_default_1, mul_345);  convert_element_type_87 = None
    mul_347: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_136, 1.1111111111111112);  where_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_595: "f32[512, 1536]" = torch.ops.aten.view.default(mul_347, [512, 1536]);  mul_347 = None
    mm_30: "f32[512, 1536]" = torch.ops.aten.mm.default(view_595, permute_416);  permute_416 = None
    permute_417: "f32[1536, 512]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_31: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_417, view_478);  permute_417 = view_478 = None
    permute_418: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_74: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[1536]" = torch.ops.aten.view.default(sum_74, [1536]);  sum_74 = None
    permute_419: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    view_597: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_30, [1, 512, 1536]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_598: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_597, [1, 512, 24, 64]);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_420: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_599: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_420, [24, 512, 64]);  permute_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_56: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_421, view_599);  permute_421 = None
    bmm_57: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_599, permute_422);  view_599 = permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_600: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 24, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_137: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_86, full_default_1, view_600);  convert_element_type_86 = view_600 = None
    mul_348: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_137, 1.1111111111111112);  where_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_349: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_348, alias_36);  mul_348 = None
    sum_75: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [-1], True)
    mul_350: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_36, sum_75);  alias_36 = sum_75 = None
    sub_172: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_601: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_172, [24, 512, 512]);  sub_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_58: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_423, view_601);  permute_423 = None
    bmm_59: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_601, permute_424);  view_601 = permute_424 = None
    div_62: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_58, full_default_2);  bmm_58 = None
    permute_425: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_62, [0, 2, 1]);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_602: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 24, 512, 64]);  bmm_56 = None
    permute_426: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_102: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_603: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_102, [1, 512, 1536]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_604: "f32[512, 1536]" = torch.ops.aten.view.default(view_603, [512, 1536]);  view_603 = None
    mm_32: "f32[512, 1536]" = torch.ops.aten.mm.default(view_604, permute_427);  permute_427 = None
    permute_428: "f32[1536, 512]" = torch.ops.aten.permute.default(view_604, [1, 0])
    mm_33: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_428, view_462);  permute_428 = None
    permute_429: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_76: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
    view_605: "f32[1536]" = torch.ops.aten.view.default(sum_76, [1536]);  sum_76 = None
    permute_430: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_606: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_32, [1, 512, 1536]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_189: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_345, view_606);  mul_345 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_607: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_425, [1, 24, 512, 64]);  permute_425 = None
    permute_431: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_608: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_431, [1, 512, 1536]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_609: "f32[512, 1536]" = torch.ops.aten.view.default(view_608, [512, 1536]);  view_608 = None
    mm_34: "f32[512, 1536]" = torch.ops.aten.mm.default(view_609, permute_432);  permute_432 = None
    permute_433: "f32[1536, 512]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_35: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_433, view_462);  permute_433 = None
    permute_434: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_77: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[1536]" = torch.ops.aten.view.default(sum_77, [1536]);  sum_77 = None
    permute_435: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_611: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_34, [1, 512, 1536]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_190: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_189, view_611);  add_189 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_612: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 24, 512, 64]);  bmm_59 = None
    permute_436: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_103: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    view_613: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_103, [1, 512, 1536]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_614: "f32[512, 1536]" = torch.ops.aten.view.default(view_613, [512, 1536]);  view_613 = None
    mm_36: "f32[512, 1536]" = torch.ops.aten.mm.default(view_614, permute_437);  permute_437 = None
    permute_438: "f32[1536, 512]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_37: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_438, view_462);  permute_438 = view_462 = None
    permute_439: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_78: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[1536]" = torch.ops.aten.view.default(sum_78, [1536]);  sum_78 = None
    permute_440: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_439, [1, 0]);  permute_439 = None
    view_616: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_36, [1, 512, 1536]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_191: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_190, view_616);  add_190 = view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_352: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_191, primals_339);  primals_339 = None
    mul_353: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_352, 1536)
    sum_79: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
    mul_354: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_352, mul_234);  mul_352 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
    mul_355: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_234, sum_80);  sum_80 = None
    sub_174: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_353, sum_79);  mul_353 = sum_79 = None
    sub_175: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_174, mul_355);  sub_174 = mul_355 = None
    mul_356: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_63, sub_175);  div_63 = sub_175 = None
    mul_357: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_191, mul_234);  mul_234 = None
    sum_81: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
    sum_82: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_191, [0, 1]);  add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_138: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_84, full_default_1, mul_356);  convert_element_type_84 = None
    mul_358: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_138, 1.1111111111111112);  where_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_617: "f32[512, 1536]" = torch.ops.aten.view.default(mul_358, [512, 1536]);  mul_358 = None
    mm_38: "f32[512, 6144]" = torch.ops.aten.mm.default(view_617, permute_441);  permute_441 = None
    permute_442: "f32[1536, 512]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_39: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_442, view_460);  permute_442 = view_460 = None
    permute_443: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_83: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[1536]" = torch.ops.aten.view.default(sum_83, [1536]);  sum_83 = None
    permute_444: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    view_619: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_38, [1, 512, 6144]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_360: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
    mul_361: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, view_459)
    mul_362: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_361, -0.5);  mul_361 = None
    exp_31: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_362);  mul_362 = None
    mul_363: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_364: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, mul_363);  view_459 = mul_363 = None
    add_193: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_360, mul_364);  mul_360 = mul_364 = None
    mul_365: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_619, add_193);  view_619 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_620: "f32[512, 6144]" = torch.ops.aten.view.default(mul_365, [512, 6144]);  mul_365 = None
    mm_40: "f32[512, 1536]" = torch.ops.aten.mm.default(view_620, permute_445);  permute_445 = None
    permute_446: "f32[6144, 512]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_41: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_446, view_458);  permute_446 = view_458 = None
    permute_447: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_620, [0], True);  view_620 = None
    view_621: "f32[6144]" = torch.ops.aten.view.default(sum_84, [6144]);  sum_84 = None
    permute_448: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    view_622: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_40, [1, 512, 1536]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_194: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_356, view_622);  mul_356 = view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_367: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_194, primals_333);  primals_333 = None
    mul_368: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_367, 1536)
    sum_85: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_367, mul_228);  mul_367 = None
    sum_86: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_228, sum_86);  sum_86 = None
    sub_177: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_368, sum_85);  mul_368 = sum_85 = None
    sub_178: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_177, mul_370);  sub_177 = mul_370 = None
    mul_371: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_64, sub_178);  div_64 = sub_178 = None
    mul_372: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_194, mul_228);  mul_228 = None
    sum_87: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_88: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_139: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_83, full_default_1, mul_371);  convert_element_type_83 = None
    mul_373: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_139, 1.1111111111111112);  where_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_623: "f32[512, 1536]" = torch.ops.aten.view.default(mul_373, [512, 1536]);  mul_373 = None
    mm_42: "f32[512, 1536]" = torch.ops.aten.mm.default(view_623, permute_449);  permute_449 = None
    permute_450: "f32[1536, 512]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_43: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_450, view_456);  permute_450 = view_456 = None
    permute_451: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_89: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_623, [0], True);  view_623 = None
    view_624: "f32[1536]" = torch.ops.aten.view.default(sum_89, [1536]);  sum_89 = None
    permute_452: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
    view_625: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_42, [1, 512, 1536]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_626: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_625, [1, 512, 24, 64]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_453: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_627: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_453, [24, 512, 64]);  permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_60: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_454, view_627);  permute_454 = None
    bmm_61: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_627, permute_455);  view_627 = permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_628: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 24, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_140: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_82, full_default_1, view_628);  convert_element_type_82 = view_628 = None
    mul_374: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_140, 1.1111111111111112);  where_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_375: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_374, alias_39);  mul_374 = None
    sum_90: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [-1], True)
    mul_376: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_90);  alias_39 = sum_90 = None
    sub_179: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_629: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_179, [24, 512, 512]);  sub_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_62: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_456, view_629);  permute_456 = None
    bmm_63: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_629, permute_457);  view_629 = permute_457 = None
    div_65: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_62, full_default_2);  bmm_62 = None
    permute_458: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_65, [0, 2, 1]);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_630: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 24, 512, 64]);  bmm_60 = None
    permute_459: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_104: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_631: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_104, [1, 512, 1536]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_632: "f32[512, 1536]" = torch.ops.aten.view.default(view_631, [512, 1536]);  view_631 = None
    mm_44: "f32[512, 1536]" = torch.ops.aten.mm.default(view_632, permute_460);  permute_460 = None
    permute_461: "f32[1536, 512]" = torch.ops.aten.permute.default(view_632, [1, 0])
    mm_45: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_461, view_440);  permute_461 = None
    permute_462: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_91: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_632, [0], True);  view_632 = None
    view_633: "f32[1536]" = torch.ops.aten.view.default(sum_91, [1536]);  sum_91 = None
    permute_463: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_634: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_44, [1, 512, 1536]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_195: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_371, view_634);  mul_371 = view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_635: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_458, [1, 24, 512, 64]);  permute_458 = None
    permute_464: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_636: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_464, [1, 512, 1536]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_637: "f32[512, 1536]" = torch.ops.aten.view.default(view_636, [512, 1536]);  view_636 = None
    mm_46: "f32[512, 1536]" = torch.ops.aten.mm.default(view_637, permute_465);  permute_465 = None
    permute_466: "f32[1536, 512]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_47: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_466, view_440);  permute_466 = None
    permute_467: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_92: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
    view_638: "f32[1536]" = torch.ops.aten.view.default(sum_92, [1536]);  sum_92 = None
    permute_468: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    view_639: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_46, [1, 512, 1536]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_196: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_195, view_639);  add_195 = view_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_640: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 24, 512, 64]);  bmm_63 = None
    permute_469: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_105: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
    view_641: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_105, [1, 512, 1536]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_642: "f32[512, 1536]" = torch.ops.aten.view.default(view_641, [512, 1536]);  view_641 = None
    mm_48: "f32[512, 1536]" = torch.ops.aten.mm.default(view_642, permute_470);  permute_470 = None
    permute_471: "f32[1536, 512]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_49: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_471, view_440);  permute_471 = view_440 = None
    permute_472: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_93: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[1536]" = torch.ops.aten.view.default(sum_93, [1536]);  sum_93 = None
    permute_473: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_644: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_48, [1, 512, 1536]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_197: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_196, view_644);  add_196 = view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_378: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_197, primals_323);  primals_323 = None
    mul_379: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_378, 1536)
    sum_94: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True)
    mul_380: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_378, mul_223);  mul_378 = None
    sum_95: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
    mul_381: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_223, sum_95);  sum_95 = None
    sub_181: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_379, sum_94);  mul_379 = sum_94 = None
    sub_182: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_181, mul_381);  sub_181 = mul_381 = None
    mul_382: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_66, sub_182);  div_66 = sub_182 = None
    mul_383: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_197, mul_223);  mul_223 = None
    sum_96: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1]);  mul_383 = None
    sum_97: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_141: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_80, full_default_1, mul_382);  convert_element_type_80 = None
    mul_384: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_141, 1.1111111111111112);  where_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_645: "f32[512, 1536]" = torch.ops.aten.view.default(mul_384, [512, 1536]);  mul_384 = None
    mm_50: "f32[512, 6144]" = torch.ops.aten.mm.default(view_645, permute_474);  permute_474 = None
    permute_475: "f32[1536, 512]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_51: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_475, view_438);  permute_475 = view_438 = None
    permute_476: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_98: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[1536]" = torch.ops.aten.view.default(sum_98, [1536]);  sum_98 = None
    permute_477: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    view_647: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_50, [1, 512, 6144]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_386: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_139, 0.5);  add_139 = None
    mul_387: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, view_437)
    mul_388: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_387, -0.5);  mul_387 = None
    exp_32: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_388);  mul_388 = None
    mul_389: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_390: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, mul_389);  view_437 = mul_389 = None
    add_199: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_386, mul_390);  mul_386 = mul_390 = None
    mul_391: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_647, add_199);  view_647 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_648: "f32[512, 6144]" = torch.ops.aten.view.default(mul_391, [512, 6144]);  mul_391 = None
    mm_52: "f32[512, 1536]" = torch.ops.aten.mm.default(view_648, permute_478);  permute_478 = None
    permute_479: "f32[6144, 512]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_53: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_479, view_436);  permute_479 = view_436 = None
    permute_480: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[6144]" = torch.ops.aten.view.default(sum_99, [6144]);  sum_99 = None
    permute_481: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    view_650: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_52, [1, 512, 1536]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_200: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_382, view_650);  mul_382 = view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_393: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_200, primals_317);  primals_317 = None
    mul_394: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_393, 1536)
    sum_100: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True)
    mul_395: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_393, mul_217);  mul_393 = None
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
    mul_396: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_217, sum_101);  sum_101 = None
    sub_184: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_394, sum_100);  mul_394 = sum_100 = None
    sub_185: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_184, mul_396);  sub_184 = mul_396 = None
    mul_397: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_67, sub_185);  div_67 = sub_185 = None
    mul_398: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_200, mul_217);  mul_217 = None
    sum_102: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1]);  mul_398 = None
    sum_103: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_142: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_79, full_default_1, mul_397);  convert_element_type_79 = None
    mul_399: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_142, 1.1111111111111112);  where_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_651: "f32[512, 1536]" = torch.ops.aten.view.default(mul_399, [512, 1536]);  mul_399 = None
    mm_54: "f32[512, 1536]" = torch.ops.aten.mm.default(view_651, permute_482);  permute_482 = None
    permute_483: "f32[1536, 512]" = torch.ops.aten.permute.default(view_651, [1, 0])
    mm_55: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_483, view_434);  permute_483 = view_434 = None
    permute_484: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_104: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_651, [0], True);  view_651 = None
    view_652: "f32[1536]" = torch.ops.aten.view.default(sum_104, [1536]);  sum_104 = None
    permute_485: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    view_653: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_54, [1, 512, 1536]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_654: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_653, [1, 512, 24, 64]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_486: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_655: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_486, [24, 512, 64]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_64: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_487, view_655);  permute_487 = None
    bmm_65: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_655, permute_488);  view_655 = permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_656: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 24, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_143: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_78, full_default_1, view_656);  convert_element_type_78 = view_656 = None
    mul_400: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_143, 1.1111111111111112);  where_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_401: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_400, alias_42);  mul_400 = None
    sum_105: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
    mul_402: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_42, sum_105);  alias_42 = sum_105 = None
    sub_186: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_657: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_186, [24, 512, 512]);  sub_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_66: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_489, view_657);  permute_489 = None
    bmm_67: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_657, permute_490);  view_657 = permute_490 = None
    div_68: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_66, full_default_2);  bmm_66 = None
    permute_491: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_68, [0, 2, 1]);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_658: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 24, 512, 64]);  bmm_64 = None
    permute_492: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_106: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_492, memory_format = torch.contiguous_format);  permute_492 = None
    view_659: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_106, [1, 512, 1536]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_660: "f32[512, 1536]" = torch.ops.aten.view.default(view_659, [512, 1536]);  view_659 = None
    mm_56: "f32[512, 1536]" = torch.ops.aten.mm.default(view_660, permute_493);  permute_493 = None
    permute_494: "f32[1536, 512]" = torch.ops.aten.permute.default(view_660, [1, 0])
    mm_57: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_494, view_418);  permute_494 = None
    permute_495: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_106: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
    view_661: "f32[1536]" = torch.ops.aten.view.default(sum_106, [1536]);  sum_106 = None
    permute_496: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_662: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_56, [1, 512, 1536]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_201: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_397, view_662);  mul_397 = view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_663: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_491, [1, 24, 512, 64]);  permute_491 = None
    permute_497: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_663, [0, 2, 1, 3]);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_664: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_497, [1, 512, 1536]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_665: "f32[512, 1536]" = torch.ops.aten.view.default(view_664, [512, 1536]);  view_664 = None
    mm_58: "f32[512, 1536]" = torch.ops.aten.mm.default(view_665, permute_498);  permute_498 = None
    permute_499: "f32[1536, 512]" = torch.ops.aten.permute.default(view_665, [1, 0])
    mm_59: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_499, view_418);  permute_499 = None
    permute_500: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_107: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_665, [0], True);  view_665 = None
    view_666: "f32[1536]" = torch.ops.aten.view.default(sum_107, [1536]);  sum_107 = None
    permute_501: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_667: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_58, [1, 512, 1536]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_202: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_201, view_667);  add_201 = view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_668: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 24, 512, 64]);  bmm_67 = None
    permute_502: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_668, [0, 2, 1, 3]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_107: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
    view_669: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_107, [1, 512, 1536]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_670: "f32[512, 1536]" = torch.ops.aten.view.default(view_669, [512, 1536]);  view_669 = None
    mm_60: "f32[512, 1536]" = torch.ops.aten.mm.default(view_670, permute_503);  permute_503 = None
    permute_504: "f32[1536, 512]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_61: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_504, view_418);  permute_504 = view_418 = None
    permute_505: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_108: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[1536]" = torch.ops.aten.view.default(sum_108, [1536]);  sum_108 = None
    permute_506: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
    view_672: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_60, [1, 512, 1536]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_203: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_202, view_672);  add_202 = view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_404: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_203, primals_307);  primals_307 = None
    mul_405: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_404, 1536)
    sum_109: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_404, mul_212);  mul_404 = None
    sum_110: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_212, sum_110);  sum_110 = None
    sub_188: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_405, sum_109);  mul_405 = sum_109 = None
    sub_189: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_188, mul_407);  sub_188 = mul_407 = None
    mul_408: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_69, sub_189);  div_69 = sub_189 = None
    mul_409: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_203, mul_212);  mul_212 = None
    sum_111: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
    sum_112: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_144: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_76, full_default_1, mul_408);  convert_element_type_76 = None
    mul_410: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_144, 1.1111111111111112);  where_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[512, 1536]" = torch.ops.aten.view.default(mul_410, [512, 1536]);  mul_410 = None
    mm_62: "f32[512, 6144]" = torch.ops.aten.mm.default(view_673, permute_507);  permute_507 = None
    permute_508: "f32[1536, 512]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_63: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_508, view_416);  permute_508 = view_416 = None
    permute_509: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_113: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
    view_674: "f32[1536]" = torch.ops.aten.view.default(sum_113, [1536]);  sum_113 = None
    permute_510: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_509, [1, 0]);  permute_509 = None
    view_675: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_62, [1, 512, 6144]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_412: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_413: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_414: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_413, -0.5);  mul_413 = None
    exp_33: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_414);  mul_414 = None
    mul_415: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_416: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, mul_415);  view_415 = mul_415 = None
    add_205: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_412, mul_416);  mul_412 = mul_416 = None
    mul_417: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_675, add_205);  view_675 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[512, 6144]" = torch.ops.aten.view.default(mul_417, [512, 6144]);  mul_417 = None
    mm_64: "f32[512, 1536]" = torch.ops.aten.mm.default(view_676, permute_511);  permute_511 = None
    permute_512: "f32[6144, 512]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_65: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_512, view_414);  permute_512 = view_414 = None
    permute_513: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_114: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
    view_677: "f32[6144]" = torch.ops.aten.view.default(sum_114, [6144]);  sum_114 = None
    permute_514: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    view_678: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_64, [1, 512, 1536]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_206: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_408, view_678);  mul_408 = view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_419: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_206, primals_301);  primals_301 = None
    mul_420: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_419, 1536)
    sum_115: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
    mul_421: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_419, mul_206);  mul_419 = None
    sum_116: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    mul_422: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_206, sum_116);  sum_116 = None
    sub_191: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_420, sum_115);  mul_420 = sum_115 = None
    sub_192: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_191, mul_422);  sub_191 = mul_422 = None
    mul_423: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_70, sub_192);  div_70 = sub_192 = None
    mul_424: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_206, mul_206);  mul_206 = None
    sum_117: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
    sum_118: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_145: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_75, full_default_1, mul_423);  convert_element_type_75 = None
    mul_425: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_145, 1.1111111111111112);  where_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_679: "f32[512, 1536]" = torch.ops.aten.view.default(mul_425, [512, 1536]);  mul_425 = None
    mm_66: "f32[512, 1536]" = torch.ops.aten.mm.default(view_679, permute_515);  permute_515 = None
    permute_516: "f32[1536, 512]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_67: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_516, view_412);  permute_516 = view_412 = None
    permute_517: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_119: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True);  view_679 = None
    view_680: "f32[1536]" = torch.ops.aten.view.default(sum_119, [1536]);  sum_119 = None
    permute_518: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    view_681: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_66, [1, 512, 1536]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_682: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_681, [1, 512, 24, 64]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_519: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_683: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_519, [24, 512, 64]);  permute_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_68: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_520, view_683);  permute_520 = None
    bmm_69: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_683, permute_521);  view_683 = permute_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_684: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 24, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_146: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_74, full_default_1, view_684);  convert_element_type_74 = view_684 = None
    mul_426: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_146, 1.1111111111111112);  where_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_427: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_426, alias_45);  mul_426 = None
    sum_120: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [-1], True)
    mul_428: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_120);  alias_45 = sum_120 = None
    sub_193: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_685: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_193, [24, 512, 512]);  sub_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_70: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_522, view_685);  permute_522 = None
    bmm_71: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_685, permute_523);  view_685 = permute_523 = None
    div_71: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_70, full_default_2);  bmm_70 = None
    permute_524: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_71, [0, 2, 1]);  div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_686: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 24, 512, 64]);  bmm_68 = None
    permute_525: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_108: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_687: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_108, [1, 512, 1536]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_688: "f32[512, 1536]" = torch.ops.aten.view.default(view_687, [512, 1536]);  view_687 = None
    mm_68: "f32[512, 1536]" = torch.ops.aten.mm.default(view_688, permute_526);  permute_526 = None
    permute_527: "f32[1536, 512]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_69: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_527, view_396);  permute_527 = None
    permute_528: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_121: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_688, [0], True);  view_688 = None
    view_689: "f32[1536]" = torch.ops.aten.view.default(sum_121, [1536]);  sum_121 = None
    permute_529: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_690: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_68, [1, 512, 1536]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_207: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_423, view_690);  mul_423 = view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_691: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_524, [1, 24, 512, 64]);  permute_524 = None
    permute_530: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_692: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_530, [1, 512, 1536]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_693: "f32[512, 1536]" = torch.ops.aten.view.default(view_692, [512, 1536]);  view_692 = None
    mm_70: "f32[512, 1536]" = torch.ops.aten.mm.default(view_693, permute_531);  permute_531 = None
    permute_532: "f32[1536, 512]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_71: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_532, view_396);  permute_532 = None
    permute_533: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
    view_694: "f32[1536]" = torch.ops.aten.view.default(sum_122, [1536]);  sum_122 = None
    permute_534: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_695: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_70, [1, 512, 1536]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_208: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_207, view_695);  add_207 = view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_696: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 24, 512, 64]);  bmm_71 = None
    permute_535: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_109: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
    view_697: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_109, [1, 512, 1536]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_698: "f32[512, 1536]" = torch.ops.aten.view.default(view_697, [512, 1536]);  view_697 = None
    mm_72: "f32[512, 1536]" = torch.ops.aten.mm.default(view_698, permute_536);  permute_536 = None
    permute_537: "f32[1536, 512]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_73: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_537, view_396);  permute_537 = view_396 = None
    permute_538: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[1536]" = torch.ops.aten.view.default(sum_123, [1536]);  sum_123 = None
    permute_539: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_700: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_72, [1, 512, 1536]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_209: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_208, view_700);  add_208 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_430: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_209, primals_291);  primals_291 = None
    mul_431: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_430, 1536)
    sum_124: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True)
    mul_432: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_430, mul_201);  mul_430 = None
    sum_125: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
    mul_433: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_201, sum_125);  sum_125 = None
    sub_195: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_431, sum_124);  mul_431 = sum_124 = None
    sub_196: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_195, mul_433);  sub_195 = mul_433 = None
    mul_434: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_72, sub_196);  div_72 = sub_196 = None
    mul_435: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_209, mul_201);  mul_201 = None
    sum_126: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
    sum_127: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_147: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_72, full_default_1, mul_434);  convert_element_type_72 = None
    mul_436: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_147, 1.1111111111111112);  where_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_701: "f32[512, 1536]" = torch.ops.aten.view.default(mul_436, [512, 1536]);  mul_436 = None
    mm_74: "f32[512, 6144]" = torch.ops.aten.mm.default(view_701, permute_540);  permute_540 = None
    permute_541: "f32[1536, 512]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_75: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_541, view_394);  permute_541 = view_394 = None
    permute_542: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_128: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[1536]" = torch.ops.aten.view.default(sum_128, [1536]);  sum_128 = None
    permute_543: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_542, [1, 0]);  permute_542 = None
    view_703: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_74, [1, 512, 6144]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_438: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_439: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_440: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_439, -0.5);  mul_439 = None
    exp_34: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_440);  mul_440 = None
    mul_441: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_442: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, mul_441);  view_393 = mul_441 = None
    add_211: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_438, mul_442);  mul_438 = mul_442 = None
    mul_443: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_703, add_211);  view_703 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_704: "f32[512, 6144]" = torch.ops.aten.view.default(mul_443, [512, 6144]);  mul_443 = None
    mm_76: "f32[512, 1536]" = torch.ops.aten.mm.default(view_704, permute_544);  permute_544 = None
    permute_545: "f32[6144, 512]" = torch.ops.aten.permute.default(view_704, [1, 0])
    mm_77: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_545, view_392);  permute_545 = view_392 = None
    permute_546: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_129: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_704, [0], True);  view_704 = None
    view_705: "f32[6144]" = torch.ops.aten.view.default(sum_129, [6144]);  sum_129 = None
    permute_547: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_706: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_76, [1, 512, 1536]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_212: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_434, view_706);  mul_434 = view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_445: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_212, primals_285);  primals_285 = None
    mul_446: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_445, 1536)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
    mul_447: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_445, mul_195);  mul_445 = None
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    mul_448: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_195, sum_131);  sum_131 = None
    sub_198: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_446, sum_130);  mul_446 = sum_130 = None
    sub_199: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_198, mul_448);  sub_198 = mul_448 = None
    mul_449: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_73, sub_199);  div_73 = sub_199 = None
    mul_450: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_212, mul_195);  mul_195 = None
    sum_132: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
    sum_133: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_148: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_71, full_default_1, mul_449);  convert_element_type_71 = None
    mul_451: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_148, 1.1111111111111112);  where_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_707: "f32[512, 1536]" = torch.ops.aten.view.default(mul_451, [512, 1536]);  mul_451 = None
    mm_78: "f32[512, 1536]" = torch.ops.aten.mm.default(view_707, permute_548);  permute_548 = None
    permute_549: "f32[1536, 512]" = torch.ops.aten.permute.default(view_707, [1, 0])
    mm_79: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_549, view_390);  permute_549 = view_390 = None
    permute_550: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_134: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_707, [0], True);  view_707 = None
    view_708: "f32[1536]" = torch.ops.aten.view.default(sum_134, [1536]);  sum_134 = None
    permute_551: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_709: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_78, [1, 512, 1536]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_710: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_709, [1, 512, 24, 64]);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_552: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_711: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_552, [24, 512, 64]);  permute_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_72: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_553, view_711);  permute_553 = None
    bmm_73: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_711, permute_554);  view_711 = permute_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_712: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_73, [1, 24, 512, 512]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_149: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_70, full_default_1, view_712);  convert_element_type_70 = view_712 = None
    mul_452: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_149, 1.1111111111111112);  where_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_453: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_452, alias_48);  mul_452 = None
    sum_135: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
    mul_454: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_135);  alias_48 = sum_135 = None
    sub_200: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_713: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_200, [24, 512, 512]);  sub_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_74: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_555, view_713);  permute_555 = None
    bmm_75: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_713, permute_556);  view_713 = permute_556 = None
    div_74: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_74, full_default_2);  bmm_74 = None
    permute_557: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_74, [0, 2, 1]);  div_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_714: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_72, [1, 24, 512, 64]);  bmm_72 = None
    permute_558: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_110: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
    view_715: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_110, [1, 512, 1536]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_716: "f32[512, 1536]" = torch.ops.aten.view.default(view_715, [512, 1536]);  view_715 = None
    mm_80: "f32[512, 1536]" = torch.ops.aten.mm.default(view_716, permute_559);  permute_559 = None
    permute_560: "f32[1536, 512]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_81: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_560, view_374);  permute_560 = None
    permute_561: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_136: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_716, [0], True);  view_716 = None
    view_717: "f32[1536]" = torch.ops.aten.view.default(sum_136, [1536]);  sum_136 = None
    permute_562: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_718: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_80, [1, 512, 1536]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_213: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_449, view_718);  mul_449 = view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_719: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_557, [1, 24, 512, 64]);  permute_557 = None
    permute_563: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_720: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_563, [1, 512, 1536]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_721: "f32[512, 1536]" = torch.ops.aten.view.default(view_720, [512, 1536]);  view_720 = None
    mm_82: "f32[512, 1536]" = torch.ops.aten.mm.default(view_721, permute_564);  permute_564 = None
    permute_565: "f32[1536, 512]" = torch.ops.aten.permute.default(view_721, [1, 0])
    mm_83: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_565, view_374);  permute_565 = None
    permute_566: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_137: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_721, [0], True);  view_721 = None
    view_722: "f32[1536]" = torch.ops.aten.view.default(sum_137, [1536]);  sum_137 = None
    permute_567: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    view_723: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_82, [1, 512, 1536]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_214: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_213, view_723);  add_213 = view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_724: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_75, [1, 24, 512, 64]);  bmm_75 = None
    permute_568: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_111: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_568, memory_format = torch.contiguous_format);  permute_568 = None
    view_725: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_111, [1, 512, 1536]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_726: "f32[512, 1536]" = torch.ops.aten.view.default(view_725, [512, 1536]);  view_725 = None
    mm_84: "f32[512, 1536]" = torch.ops.aten.mm.default(view_726, permute_569);  permute_569 = None
    permute_570: "f32[1536, 512]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_85: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_570, view_374);  permute_570 = view_374 = None
    permute_571: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_138: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[1536]" = torch.ops.aten.view.default(sum_138, [1536]);  sum_138 = None
    permute_572: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_571, [1, 0]);  permute_571 = None
    view_728: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_84, [1, 512, 1536]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_215: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_214, view_728);  add_214 = view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_456: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_215, primals_275);  primals_275 = None
    mul_457: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_456, 1536)
    sum_139: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True)
    mul_458: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_456, mul_190);  mul_456 = None
    sum_140: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True);  mul_458 = None
    mul_459: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_190, sum_140);  sum_140 = None
    sub_202: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_457, sum_139);  mul_457 = sum_139 = None
    sub_203: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_202, mul_459);  sub_202 = mul_459 = None
    mul_460: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_75, sub_203);  div_75 = sub_203 = None
    mul_461: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_215, mul_190);  mul_190 = None
    sum_141: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1]);  mul_461 = None
    sum_142: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_150: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_68, full_default_1, mul_460);  convert_element_type_68 = None
    mul_462: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_150, 1.1111111111111112);  where_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_729: "f32[512, 1536]" = torch.ops.aten.view.default(mul_462, [512, 1536]);  mul_462 = None
    mm_86: "f32[512, 6144]" = torch.ops.aten.mm.default(view_729, permute_573);  permute_573 = None
    permute_574: "f32[1536, 512]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_87: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_574, view_372);  permute_574 = view_372 = None
    permute_575: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_143: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[1536]" = torch.ops.aten.view.default(sum_143, [1536]);  sum_143 = None
    permute_576: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
    view_731: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_86, [1, 512, 6144]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_464: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
    mul_465: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_466: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_465, -0.5);  mul_465 = None
    exp_35: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_466);  mul_466 = None
    mul_467: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_468: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, mul_467);  view_371 = mul_467 = None
    add_217: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_464, mul_468);  mul_464 = mul_468 = None
    mul_469: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_731, add_217);  view_731 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_732: "f32[512, 6144]" = torch.ops.aten.view.default(mul_469, [512, 6144]);  mul_469 = None
    mm_88: "f32[512, 1536]" = torch.ops.aten.mm.default(view_732, permute_577);  permute_577 = None
    permute_578: "f32[6144, 512]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_89: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_578, view_370);  permute_578 = view_370 = None
    permute_579: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_144: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_732, [0], True);  view_732 = None
    view_733: "f32[6144]" = torch.ops.aten.view.default(sum_144, [6144]);  sum_144 = None
    permute_580: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_734: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_88, [1, 512, 1536]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_218: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_460, view_734);  mul_460 = view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_471: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_218, primals_269);  primals_269 = None
    mul_472: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_471, 1536)
    sum_145: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
    mul_473: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_471, mul_184);  mul_471 = None
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    mul_474: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_184, sum_146);  sum_146 = None
    sub_205: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_472, sum_145);  mul_472 = sum_145 = None
    sub_206: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_205, mul_474);  sub_205 = mul_474 = None
    mul_475: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_76, sub_206);  div_76 = sub_206 = None
    mul_476: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_218, mul_184);  mul_184 = None
    sum_147: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    sum_148: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_151: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_67, full_default_1, mul_475);  convert_element_type_67 = None
    mul_477: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_151, 1.1111111111111112);  where_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_735: "f32[512, 1536]" = torch.ops.aten.view.default(mul_477, [512, 1536]);  mul_477 = None
    mm_90: "f32[512, 1536]" = torch.ops.aten.mm.default(view_735, permute_581);  permute_581 = None
    permute_582: "f32[1536, 512]" = torch.ops.aten.permute.default(view_735, [1, 0])
    mm_91: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_582, view_368);  permute_582 = view_368 = None
    permute_583: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_149: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_735, [0], True);  view_735 = None
    view_736: "f32[1536]" = torch.ops.aten.view.default(sum_149, [1536]);  sum_149 = None
    permute_584: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_737: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_90, [1, 512, 1536]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_738: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_737, [1, 512, 24, 64]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_585: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_739: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_585, [24, 512, 64]);  permute_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_76: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_586, view_739);  permute_586 = None
    bmm_77: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_739, permute_587);  view_739 = permute_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_740: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_77, [1, 24, 512, 512]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_152: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_66, full_default_1, view_740);  convert_element_type_66 = view_740 = None
    mul_478: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_152, 1.1111111111111112);  where_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_479: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_478, alias_51);  mul_478 = None
    sum_150: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [-1], True)
    mul_480: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_51, sum_150);  alias_51 = sum_150 = None
    sub_207: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_741: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_207, [24, 512, 512]);  sub_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_78: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_588, view_741);  permute_588 = None
    bmm_79: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_741, permute_589);  view_741 = permute_589 = None
    div_77: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_78, full_default_2);  bmm_78 = None
    permute_590: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_77, [0, 2, 1]);  div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_742: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_76, [1, 24, 512, 64]);  bmm_76 = None
    permute_591: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_112: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_591, memory_format = torch.contiguous_format);  permute_591 = None
    view_743: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_112, [1, 512, 1536]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_744: "f32[512, 1536]" = torch.ops.aten.view.default(view_743, [512, 1536]);  view_743 = None
    mm_92: "f32[512, 1536]" = torch.ops.aten.mm.default(view_744, permute_592);  permute_592 = None
    permute_593: "f32[1536, 512]" = torch.ops.aten.permute.default(view_744, [1, 0])
    mm_93: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_593, view_352);  permute_593 = None
    permute_594: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_151: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_744, [0], True);  view_744 = None
    view_745: "f32[1536]" = torch.ops.aten.view.default(sum_151, [1536]);  sum_151 = None
    permute_595: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_746: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_92, [1, 512, 1536]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_219: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_475, view_746);  mul_475 = view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_747: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_590, [1, 24, 512, 64]);  permute_590 = None
    permute_596: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_747, [0, 2, 1, 3]);  view_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_748: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_596, [1, 512, 1536]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_749: "f32[512, 1536]" = torch.ops.aten.view.default(view_748, [512, 1536]);  view_748 = None
    mm_94: "f32[512, 1536]" = torch.ops.aten.mm.default(view_749, permute_597);  permute_597 = None
    permute_598: "f32[1536, 512]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_95: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_598, view_352);  permute_598 = None
    permute_599: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_152: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[1536]" = torch.ops.aten.view.default(sum_152, [1536]);  sum_152 = None
    permute_600: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    view_751: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_94, [1, 512, 1536]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_220: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_219, view_751);  add_219 = view_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_752: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_79, [1, 24, 512, 64]);  bmm_79 = None
    permute_601: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_113: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_601, memory_format = torch.contiguous_format);  permute_601 = None
    view_753: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_113, [1, 512, 1536]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_754: "f32[512, 1536]" = torch.ops.aten.view.default(view_753, [512, 1536]);  view_753 = None
    mm_96: "f32[512, 1536]" = torch.ops.aten.mm.default(view_754, permute_602);  permute_602 = None
    permute_603: "f32[1536, 512]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_97: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_603, view_352);  permute_603 = view_352 = None
    permute_604: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_153: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
    view_755: "f32[1536]" = torch.ops.aten.view.default(sum_153, [1536]);  sum_153 = None
    permute_605: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_604, [1, 0]);  permute_604 = None
    view_756: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_96, [1, 512, 1536]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_221: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_220, view_756);  add_220 = view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_482: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_221, primals_259);  primals_259 = None
    mul_483: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_482, 1536)
    sum_154: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_482, mul_179);  mul_482 = None
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_179, sum_155);  sum_155 = None
    sub_209: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_483, sum_154);  mul_483 = sum_154 = None
    sub_210: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_209, mul_485);  sub_209 = mul_485 = None
    mul_486: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_78, sub_210);  div_78 = sub_210 = None
    mul_487: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_221, mul_179);  mul_179 = None
    sum_156: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_157: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_153: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_64, full_default_1, mul_486);  convert_element_type_64 = None
    mul_488: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_153, 1.1111111111111112);  where_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_757: "f32[512, 1536]" = torch.ops.aten.view.default(mul_488, [512, 1536]);  mul_488 = None
    mm_98: "f32[512, 6144]" = torch.ops.aten.mm.default(view_757, permute_606);  permute_606 = None
    permute_607: "f32[1536, 512]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_99: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_607, view_350);  permute_607 = view_350 = None
    permute_608: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_158: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
    view_758: "f32[1536]" = torch.ops.aten.view.default(sum_158, [1536]);  sum_158 = None
    permute_609: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_608, [1, 0]);  permute_608 = None
    view_759: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_98, [1, 512, 6144]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_490: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
    mul_491: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, view_349)
    mul_492: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_491, -0.5);  mul_491 = None
    exp_36: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_492);  mul_492 = None
    mul_493: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_494: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, mul_493);  view_349 = mul_493 = None
    add_223: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_490, mul_494);  mul_490 = mul_494 = None
    mul_495: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_759, add_223);  view_759 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_760: "f32[512, 6144]" = torch.ops.aten.view.default(mul_495, [512, 6144]);  mul_495 = None
    mm_100: "f32[512, 1536]" = torch.ops.aten.mm.default(view_760, permute_610);  permute_610 = None
    permute_611: "f32[6144, 512]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_101: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_611, view_348);  permute_611 = view_348 = None
    permute_612: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_159: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True);  view_760 = None
    view_761: "f32[6144]" = torch.ops.aten.view.default(sum_159, [6144]);  sum_159 = None
    permute_613: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_762: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_100, [1, 512, 1536]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_224: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_486, view_762);  mul_486 = view_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_497: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_224, primals_253);  primals_253 = None
    mul_498: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_497, 1536)
    sum_160: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True)
    mul_499: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_497, mul_173);  mul_497 = None
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True);  mul_499 = None
    mul_500: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_173, sum_161);  sum_161 = None
    sub_212: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_498, sum_160);  mul_498 = sum_160 = None
    sub_213: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_212, mul_500);  sub_212 = mul_500 = None
    mul_501: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_79, sub_213);  div_79 = sub_213 = None
    mul_502: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_224, mul_173);  mul_173 = None
    sum_162: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 1]);  mul_502 = None
    sum_163: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_154: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_63, full_default_1, mul_501);  convert_element_type_63 = None
    mul_503: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_154, 1.1111111111111112);  where_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_763: "f32[512, 1536]" = torch.ops.aten.view.default(mul_503, [512, 1536]);  mul_503 = None
    mm_102: "f32[512, 1536]" = torch.ops.aten.mm.default(view_763, permute_614);  permute_614 = None
    permute_615: "f32[1536, 512]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_103: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_615, view_346);  permute_615 = view_346 = None
    permute_616: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_164: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[1536]" = torch.ops.aten.view.default(sum_164, [1536]);  sum_164 = None
    permute_617: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_616, [1, 0]);  permute_616 = None
    view_765: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_102, [1, 512, 1536]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_766: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_765, [1, 512, 24, 64]);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_618: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_767: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_618, [24, 512, 64]);  permute_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_80: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_619, view_767);  permute_619 = None
    bmm_81: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_767, permute_620);  view_767 = permute_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_768: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_81, [1, 24, 512, 512]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_155: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_62, full_default_1, view_768);  convert_element_type_62 = view_768 = None
    mul_504: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_155, 1.1111111111111112);  where_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_505: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_504, alias_54);  mul_504 = None
    sum_165: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [-1], True)
    mul_506: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_54, sum_165);  alias_54 = sum_165 = None
    sub_214: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_769: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_214, [24, 512, 512]);  sub_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_82: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_621, view_769);  permute_621 = None
    bmm_83: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_769, permute_622);  view_769 = permute_622 = None
    div_80: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_82, full_default_2);  bmm_82 = None
    permute_623: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_80, [0, 2, 1]);  div_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_770: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_80, [1, 24, 512, 64]);  bmm_80 = None
    permute_624: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_114: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_624, memory_format = torch.contiguous_format);  permute_624 = None
    view_771: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_114, [1, 512, 1536]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_772: "f32[512, 1536]" = torch.ops.aten.view.default(view_771, [512, 1536]);  view_771 = None
    mm_104: "f32[512, 1536]" = torch.ops.aten.mm.default(view_772, permute_625);  permute_625 = None
    permute_626: "f32[1536, 512]" = torch.ops.aten.permute.default(view_772, [1, 0])
    mm_105: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_626, view_330);  permute_626 = None
    permute_627: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_166: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_772, [0], True);  view_772 = None
    view_773: "f32[1536]" = torch.ops.aten.view.default(sum_166, [1536]);  sum_166 = None
    permute_628: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_774: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_104, [1, 512, 1536]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_225: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_501, view_774);  mul_501 = view_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_775: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_623, [1, 24, 512, 64]);  permute_623 = None
    permute_629: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_775, [0, 2, 1, 3]);  view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_776: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_629, [1, 512, 1536]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_777: "f32[512, 1536]" = torch.ops.aten.view.default(view_776, [512, 1536]);  view_776 = None
    mm_106: "f32[512, 1536]" = torch.ops.aten.mm.default(view_777, permute_630);  permute_630 = None
    permute_631: "f32[1536, 512]" = torch.ops.aten.permute.default(view_777, [1, 0])
    mm_107: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_631, view_330);  permute_631 = None
    permute_632: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_167: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_777, [0], True);  view_777 = None
    view_778: "f32[1536]" = torch.ops.aten.view.default(sum_167, [1536]);  sum_167 = None
    permute_633: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    view_779: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_106, [1, 512, 1536]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_226: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_225, view_779);  add_225 = view_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_780: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_83, [1, 24, 512, 64]);  bmm_83 = None
    permute_634: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_115: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
    view_781: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_115, [1, 512, 1536]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_782: "f32[512, 1536]" = torch.ops.aten.view.default(view_781, [512, 1536]);  view_781 = None
    mm_108: "f32[512, 1536]" = torch.ops.aten.mm.default(view_782, permute_635);  permute_635 = None
    permute_636: "f32[1536, 512]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_109: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_636, view_330);  permute_636 = view_330 = None
    permute_637: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_168: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
    view_783: "f32[1536]" = torch.ops.aten.view.default(sum_168, [1536]);  sum_168 = None
    permute_638: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
    view_784: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_108, [1, 512, 1536]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_227: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_226, view_784);  add_226 = view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_508: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_227, primals_243);  primals_243 = None
    mul_509: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_508, 1536)
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True)
    mul_510: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_508, mul_168);  mul_508 = None
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True);  mul_510 = None
    mul_511: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_168, sum_170);  sum_170 = None
    sub_216: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_509, sum_169);  mul_509 = sum_169 = None
    sub_217: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_216, mul_511);  sub_216 = mul_511 = None
    mul_512: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_81, sub_217);  div_81 = sub_217 = None
    mul_513: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_227, mul_168);  mul_168 = None
    sum_171: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 1]);  mul_513 = None
    sum_172: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_156: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_60, full_default_1, mul_512);  convert_element_type_60 = None
    mul_514: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_156, 1.1111111111111112);  where_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_785: "f32[512, 1536]" = torch.ops.aten.view.default(mul_514, [512, 1536]);  mul_514 = None
    mm_110: "f32[512, 6144]" = torch.ops.aten.mm.default(view_785, permute_639);  permute_639 = None
    permute_640: "f32[1536, 512]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_111: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_640, view_328);  permute_640 = view_328 = None
    permute_641: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_173: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[1536]" = torch.ops.aten.view.default(sum_173, [1536]);  sum_173 = None
    permute_642: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_641, [1, 0]);  permute_641 = None
    view_787: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_110, [1, 512, 6144]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_516: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_517: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_518: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_517, -0.5);  mul_517 = None
    exp_37: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_518);  mul_518 = None
    mul_519: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_520: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, mul_519);  view_327 = mul_519 = None
    add_229: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_516, mul_520);  mul_516 = mul_520 = None
    mul_521: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_787, add_229);  view_787 = add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_788: "f32[512, 6144]" = torch.ops.aten.view.default(mul_521, [512, 6144]);  mul_521 = None
    mm_112: "f32[512, 1536]" = torch.ops.aten.mm.default(view_788, permute_643);  permute_643 = None
    permute_644: "f32[6144, 512]" = torch.ops.aten.permute.default(view_788, [1, 0])
    mm_113: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_644, view_326);  permute_644 = view_326 = None
    permute_645: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_174: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_788, [0], True);  view_788 = None
    view_789: "f32[6144]" = torch.ops.aten.view.default(sum_174, [6144]);  sum_174 = None
    permute_646: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_645, [1, 0]);  permute_645 = None
    view_790: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_112, [1, 512, 1536]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_230: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_512, view_790);  mul_512 = view_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_523: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_230, primals_237);  primals_237 = None
    mul_524: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_523, 1536)
    sum_175: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
    mul_525: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_523, mul_162);  mul_523 = None
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
    mul_526: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_162, sum_176);  sum_176 = None
    sub_219: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_524, sum_175);  mul_524 = sum_175 = None
    sub_220: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_219, mul_526);  sub_219 = mul_526 = None
    mul_527: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_82, sub_220);  div_82 = sub_220 = None
    mul_528: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_230, mul_162);  mul_162 = None
    sum_177: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
    sum_178: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_157: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_59, full_default_1, mul_527);  convert_element_type_59 = None
    mul_529: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_157, 1.1111111111111112);  where_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_791: "f32[512, 1536]" = torch.ops.aten.view.default(mul_529, [512, 1536]);  mul_529 = None
    mm_114: "f32[512, 1536]" = torch.ops.aten.mm.default(view_791, permute_647);  permute_647 = None
    permute_648: "f32[1536, 512]" = torch.ops.aten.permute.default(view_791, [1, 0])
    mm_115: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_648, view_324);  permute_648 = view_324 = None
    permute_649: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_179: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_791, [0], True);  view_791 = None
    view_792: "f32[1536]" = torch.ops.aten.view.default(sum_179, [1536]);  sum_179 = None
    permute_650: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
    view_793: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_114, [1, 512, 1536]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_794: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_793, [1, 512, 24, 64]);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_651: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_795: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_651, [24, 512, 64]);  permute_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_84: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_652, view_795);  permute_652 = None
    bmm_85: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_795, permute_653);  view_795 = permute_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_796: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_85, [1, 24, 512, 512]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_158: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_58, full_default_1, view_796);  convert_element_type_58 = view_796 = None
    mul_530: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_158, 1.1111111111111112);  where_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_531: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_530, alias_57);  mul_530 = None
    sum_180: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [-1], True)
    mul_532: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_57, sum_180);  alias_57 = sum_180 = None
    sub_221: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_797: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_221, [24, 512, 512]);  sub_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_86: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_654, view_797);  permute_654 = None
    bmm_87: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_797, permute_655);  view_797 = permute_655 = None
    div_83: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_86, full_default_2);  bmm_86 = None
    permute_656: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_83, [0, 2, 1]);  div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_798: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_84, [1, 24, 512, 64]);  bmm_84 = None
    permute_657: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_116: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_657, memory_format = torch.contiguous_format);  permute_657 = None
    view_799: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_116, [1, 512, 1536]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_800: "f32[512, 1536]" = torch.ops.aten.view.default(view_799, [512, 1536]);  view_799 = None
    mm_116: "f32[512, 1536]" = torch.ops.aten.mm.default(view_800, permute_658);  permute_658 = None
    permute_659: "f32[1536, 512]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_117: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_659, view_308);  permute_659 = None
    permute_660: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_181: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_800, [0], True);  view_800 = None
    view_801: "f32[1536]" = torch.ops.aten.view.default(sum_181, [1536]);  sum_181 = None
    permute_661: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_660, [1, 0]);  permute_660 = None
    view_802: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_116, [1, 512, 1536]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_231: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_527, view_802);  mul_527 = view_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_803: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_656, [1, 24, 512, 64]);  permute_656 = None
    permute_662: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_803, [0, 2, 1, 3]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_804: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_662, [1, 512, 1536]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_805: "f32[512, 1536]" = torch.ops.aten.view.default(view_804, [512, 1536]);  view_804 = None
    mm_118: "f32[512, 1536]" = torch.ops.aten.mm.default(view_805, permute_663);  permute_663 = None
    permute_664: "f32[1536, 512]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_119: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_664, view_308);  permute_664 = None
    permute_665: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_182: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
    view_806: "f32[1536]" = torch.ops.aten.view.default(sum_182, [1536]);  sum_182 = None
    permute_666: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    view_807: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_118, [1, 512, 1536]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_232: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_231, view_807);  add_231 = view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_808: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_87, [1, 24, 512, 64]);  bmm_87 = None
    permute_667: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_117: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
    view_809: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_117, [1, 512, 1536]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_810: "f32[512, 1536]" = torch.ops.aten.view.default(view_809, [512, 1536]);  view_809 = None
    mm_120: "f32[512, 1536]" = torch.ops.aten.mm.default(view_810, permute_668);  permute_668 = None
    permute_669: "f32[1536, 512]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_121: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_669, view_308);  permute_669 = view_308 = None
    permute_670: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[1536]" = torch.ops.aten.view.default(sum_183, [1536]);  sum_183 = None
    permute_671: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_670, [1, 0]);  permute_670 = None
    view_812: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_120, [1, 512, 1536]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_233: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_232, view_812);  add_232 = view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_534: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_233, primals_227);  primals_227 = None
    mul_535: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_534, 1536)
    sum_184: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True)
    mul_536: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_534, mul_157);  mul_534 = None
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True);  mul_536 = None
    mul_537: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_157, sum_185);  sum_185 = None
    sub_223: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_535, sum_184);  mul_535 = sum_184 = None
    sub_224: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_223, mul_537);  sub_223 = mul_537 = None
    mul_538: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_84, sub_224);  div_84 = sub_224 = None
    mul_539: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_233, mul_157);  mul_157 = None
    sum_186: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 1]);  mul_539 = None
    sum_187: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_159: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_56, full_default_1, mul_538);  convert_element_type_56 = None
    mul_540: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_159, 1.1111111111111112);  where_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_813: "f32[512, 1536]" = torch.ops.aten.view.default(mul_540, [512, 1536]);  mul_540 = None
    mm_122: "f32[512, 6144]" = torch.ops.aten.mm.default(view_813, permute_672);  permute_672 = None
    permute_673: "f32[1536, 512]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_123: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_673, view_306);  permute_673 = view_306 = None
    permute_674: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_188: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
    view_814: "f32[1536]" = torch.ops.aten.view.default(sum_188, [1536]);  sum_188 = None
    permute_675: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None
    view_815: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_122, [1, 512, 6144]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_542: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_543: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, view_305)
    mul_544: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_543, -0.5);  mul_543 = None
    exp_38: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_544);  mul_544 = None
    mul_545: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_546: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, mul_545);  view_305 = mul_545 = None
    add_235: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_542, mul_546);  mul_542 = mul_546 = None
    mul_547: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_815, add_235);  view_815 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_816: "f32[512, 6144]" = torch.ops.aten.view.default(mul_547, [512, 6144]);  mul_547 = None
    mm_124: "f32[512, 1536]" = torch.ops.aten.mm.default(view_816, permute_676);  permute_676 = None
    permute_677: "f32[6144, 512]" = torch.ops.aten.permute.default(view_816, [1, 0])
    mm_125: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_677, view_304);  permute_677 = view_304 = None
    permute_678: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_189: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_816, [0], True);  view_816 = None
    view_817: "f32[6144]" = torch.ops.aten.view.default(sum_189, [6144]);  sum_189 = None
    permute_679: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_678, [1, 0]);  permute_678 = None
    view_818: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_124, [1, 512, 1536]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_236: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_538, view_818);  mul_538 = view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_549: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_236, primals_221);  primals_221 = None
    mul_550: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_549, 1536)
    sum_190: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True)
    mul_551: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_549, mul_151);  mul_549 = None
    sum_191: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True);  mul_551 = None
    mul_552: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_151, sum_191);  sum_191 = None
    sub_226: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_550, sum_190);  mul_550 = sum_190 = None
    sub_227: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_226, mul_552);  sub_226 = mul_552 = None
    mul_553: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_85, sub_227);  div_85 = sub_227 = None
    mul_554: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_236, mul_151);  mul_151 = None
    sum_192: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1]);  mul_554 = None
    sum_193: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_160: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_55, full_default_1, mul_553);  convert_element_type_55 = None
    mul_555: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_160, 1.1111111111111112);  where_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_819: "f32[512, 1536]" = torch.ops.aten.view.default(mul_555, [512, 1536]);  mul_555 = None
    mm_126: "f32[512, 1536]" = torch.ops.aten.mm.default(view_819, permute_680);  permute_680 = None
    permute_681: "f32[1536, 512]" = torch.ops.aten.permute.default(view_819, [1, 0])
    mm_127: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_681, view_302);  permute_681 = view_302 = None
    permute_682: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_194: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_819, [0], True);  view_819 = None
    view_820: "f32[1536]" = torch.ops.aten.view.default(sum_194, [1536]);  sum_194 = None
    permute_683: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    view_821: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_126, [1, 512, 1536]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_822: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_821, [1, 512, 24, 64]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_684: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_822, [0, 2, 1, 3]);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_823: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_684, [24, 512, 64]);  permute_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_88: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_685, view_823);  permute_685 = None
    bmm_89: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_823, permute_686);  view_823 = permute_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_824: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_89, [1, 24, 512, 512]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_161: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_54, full_default_1, view_824);  convert_element_type_54 = view_824 = None
    mul_556: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_161, 1.1111111111111112);  where_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_557: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_556, alias_60);  mul_556 = None
    sum_195: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [-1], True)
    mul_558: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_60, sum_195);  alias_60 = sum_195 = None
    sub_228: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_825: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_228, [24, 512, 512]);  sub_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_90: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_687, view_825);  permute_687 = None
    bmm_91: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_825, permute_688);  view_825 = permute_688 = None
    div_86: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_90, full_default_2);  bmm_90 = None
    permute_689: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_86, [0, 2, 1]);  div_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_826: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_88, [1, 24, 512, 64]);  bmm_88 = None
    permute_690: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_826, [0, 2, 1, 3]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_118: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
    view_827: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_118, [1, 512, 1536]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_828: "f32[512, 1536]" = torch.ops.aten.view.default(view_827, [512, 1536]);  view_827 = None
    mm_128: "f32[512, 1536]" = torch.ops.aten.mm.default(view_828, permute_691);  permute_691 = None
    permute_692: "f32[1536, 512]" = torch.ops.aten.permute.default(view_828, [1, 0])
    mm_129: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_692, view_286);  permute_692 = None
    permute_693: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_196: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
    view_829: "f32[1536]" = torch.ops.aten.view.default(sum_196, [1536]);  sum_196 = None
    permute_694: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_693, [1, 0]);  permute_693 = None
    view_830: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_128, [1, 512, 1536]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_237: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_553, view_830);  mul_553 = view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_831: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_689, [1, 24, 512, 64]);  permute_689 = None
    permute_695: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_832: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_695, [1, 512, 1536]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_833: "f32[512, 1536]" = torch.ops.aten.view.default(view_832, [512, 1536]);  view_832 = None
    mm_130: "f32[512, 1536]" = torch.ops.aten.mm.default(view_833, permute_696);  permute_696 = None
    permute_697: "f32[1536, 512]" = torch.ops.aten.permute.default(view_833, [1, 0])
    mm_131: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_697, view_286);  permute_697 = None
    permute_698: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_197: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_833, [0], True);  view_833 = None
    view_834: "f32[1536]" = torch.ops.aten.view.default(sum_197, [1536]);  sum_197 = None
    permute_699: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_835: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_130, [1, 512, 1536]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_238: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_237, view_835);  add_237 = view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_836: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_91, [1, 24, 512, 64]);  bmm_91 = None
    permute_700: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_119: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_700, memory_format = torch.contiguous_format);  permute_700 = None
    view_837: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_119, [1, 512, 1536]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_838: "f32[512, 1536]" = torch.ops.aten.view.default(view_837, [512, 1536]);  view_837 = None
    mm_132: "f32[512, 1536]" = torch.ops.aten.mm.default(view_838, permute_701);  permute_701 = None
    permute_702: "f32[1536, 512]" = torch.ops.aten.permute.default(view_838, [1, 0])
    mm_133: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_702, view_286);  permute_702 = view_286 = None
    permute_703: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_198: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
    view_839: "f32[1536]" = torch.ops.aten.view.default(sum_198, [1536]);  sum_198 = None
    permute_704: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_703, [1, 0]);  permute_703 = None
    view_840: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_132, [1, 512, 1536]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_239: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_238, view_840);  add_238 = view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_560: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_239, primals_211);  primals_211 = None
    mul_561: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_560, 1536)
    sum_199: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True)
    mul_562: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_560, mul_146);  mul_560 = None
    sum_200: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [2], True);  mul_562 = None
    mul_563: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_146, sum_200);  sum_200 = None
    sub_230: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_561, sum_199);  mul_561 = sum_199 = None
    sub_231: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_230, mul_563);  sub_230 = mul_563 = None
    mul_564: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_87, sub_231);  div_87 = sub_231 = None
    mul_565: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_239, mul_146);  mul_146 = None
    sum_201: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1]);  mul_565 = None
    sum_202: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_162: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_52, full_default_1, mul_564);  convert_element_type_52 = None
    mul_566: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_162, 1.1111111111111112);  where_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_841: "f32[512, 1536]" = torch.ops.aten.view.default(mul_566, [512, 1536]);  mul_566 = None
    mm_134: "f32[512, 6144]" = torch.ops.aten.mm.default(view_841, permute_705);  permute_705 = None
    permute_706: "f32[1536, 512]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_135: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_706, view_284);  permute_706 = view_284 = None
    permute_707: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_203: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
    view_842: "f32[1536]" = torch.ops.aten.view.default(sum_203, [1536]);  sum_203 = None
    permute_708: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_707, [1, 0]);  permute_707 = None
    view_843: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_134, [1, 512, 6144]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_568: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
    mul_569: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, view_283)
    mul_570: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_569, -0.5);  mul_569 = None
    exp_39: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_570);  mul_570 = None
    mul_571: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_572: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, mul_571);  view_283 = mul_571 = None
    add_241: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_568, mul_572);  mul_568 = mul_572 = None
    mul_573: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_843, add_241);  view_843 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_844: "f32[512, 6144]" = torch.ops.aten.view.default(mul_573, [512, 6144]);  mul_573 = None
    mm_136: "f32[512, 1536]" = torch.ops.aten.mm.default(view_844, permute_709);  permute_709 = None
    permute_710: "f32[6144, 512]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_137: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_710, view_282);  permute_710 = view_282 = None
    permute_711: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_204: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
    view_845: "f32[6144]" = torch.ops.aten.view.default(sum_204, [6144]);  sum_204 = None
    permute_712: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_711, [1, 0]);  permute_711 = None
    view_846: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_136, [1, 512, 1536]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_242: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_564, view_846);  mul_564 = view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_575: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_242, primals_205);  primals_205 = None
    mul_576: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_575, 1536)
    sum_205: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_575, [2], True)
    mul_577: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_575, mul_140);  mul_575 = None
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2], True);  mul_577 = None
    mul_578: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_140, sum_206);  sum_206 = None
    sub_233: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_576, sum_205);  mul_576 = sum_205 = None
    sub_234: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_233, mul_578);  sub_233 = mul_578 = None
    mul_579: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_88, sub_234);  div_88 = sub_234 = None
    mul_580: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_242, mul_140);  mul_140 = None
    sum_207: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 1]);  mul_580 = None
    sum_208: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_163: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_51, full_default_1, mul_579);  convert_element_type_51 = None
    mul_581: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_163, 1.1111111111111112);  where_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_847: "f32[512, 1536]" = torch.ops.aten.view.default(mul_581, [512, 1536]);  mul_581 = None
    mm_138: "f32[512, 1536]" = torch.ops.aten.mm.default(view_847, permute_713);  permute_713 = None
    permute_714: "f32[1536, 512]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_139: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_714, view_280);  permute_714 = view_280 = None
    permute_715: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_209: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[1536]" = torch.ops.aten.view.default(sum_209, [1536]);  sum_209 = None
    permute_716: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    view_849: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_138, [1, 512, 1536]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_850: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_849, [1, 512, 24, 64]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_717: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_850, [0, 2, 1, 3]);  view_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_851: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_717, [24, 512, 64]);  permute_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_92: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_718, view_851);  permute_718 = None
    bmm_93: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_851, permute_719);  view_851 = permute_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_852: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_93, [1, 24, 512, 512]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_164: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_50, full_default_1, view_852);  convert_element_type_50 = view_852 = None
    mul_582: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_164, 1.1111111111111112);  where_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_583: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_582, alias_63);  mul_582 = None
    sum_210: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [-1], True)
    mul_584: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_63, sum_210);  alias_63 = sum_210 = None
    sub_235: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_853: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_235, [24, 512, 512]);  sub_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_94: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_720, view_853);  permute_720 = None
    bmm_95: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_853, permute_721);  view_853 = permute_721 = None
    div_89: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_94, full_default_2);  bmm_94 = None
    permute_722: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_89, [0, 2, 1]);  div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_854: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_92, [1, 24, 512, 64]);  bmm_92 = None
    permute_723: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_120: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_723, memory_format = torch.contiguous_format);  permute_723 = None
    view_855: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_120, [1, 512, 1536]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_856: "f32[512, 1536]" = torch.ops.aten.view.default(view_855, [512, 1536]);  view_855 = None
    mm_140: "f32[512, 1536]" = torch.ops.aten.mm.default(view_856, permute_724);  permute_724 = None
    permute_725: "f32[1536, 512]" = torch.ops.aten.permute.default(view_856, [1, 0])
    mm_141: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_725, view_264);  permute_725 = None
    permute_726: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_211: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_856, [0], True);  view_856 = None
    view_857: "f32[1536]" = torch.ops.aten.view.default(sum_211, [1536]);  sum_211 = None
    permute_727: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_726, [1, 0]);  permute_726 = None
    view_858: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_140, [1, 512, 1536]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_243: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_579, view_858);  mul_579 = view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_859: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_722, [1, 24, 512, 64]);  permute_722 = None
    permute_728: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_859, [0, 2, 1, 3]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_860: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_728, [1, 512, 1536]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_861: "f32[512, 1536]" = torch.ops.aten.view.default(view_860, [512, 1536]);  view_860 = None
    mm_142: "f32[512, 1536]" = torch.ops.aten.mm.default(view_861, permute_729);  permute_729 = None
    permute_730: "f32[1536, 512]" = torch.ops.aten.permute.default(view_861, [1, 0])
    mm_143: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_730, view_264);  permute_730 = None
    permute_731: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_212: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_861, [0], True);  view_861 = None
    view_862: "f32[1536]" = torch.ops.aten.view.default(sum_212, [1536]);  sum_212 = None
    permute_732: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_863: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_142, [1, 512, 1536]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_244: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_243, view_863);  add_243 = view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_864: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_95, [1, 24, 512, 64]);  bmm_95 = None
    permute_733: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_864, [0, 2, 1, 3]);  view_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_121: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_733, memory_format = torch.contiguous_format);  permute_733 = None
    view_865: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_121, [1, 512, 1536]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_866: "f32[512, 1536]" = torch.ops.aten.view.default(view_865, [512, 1536]);  view_865 = None
    mm_144: "f32[512, 1536]" = torch.ops.aten.mm.default(view_866, permute_734);  permute_734 = None
    permute_735: "f32[1536, 512]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_145: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_735, view_264);  permute_735 = view_264 = None
    permute_736: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_213: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
    view_867: "f32[1536]" = torch.ops.aten.view.default(sum_213, [1536]);  sum_213 = None
    permute_737: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_736, [1, 0]);  permute_736 = None
    view_868: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_144, [1, 512, 1536]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_245: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_244, view_868);  add_244 = view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_586: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_245, primals_195);  primals_195 = None
    mul_587: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_586, 1536)
    sum_214: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True)
    mul_588: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_586, mul_135);  mul_586 = None
    sum_215: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True);  mul_588 = None
    mul_589: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_135, sum_215);  sum_215 = None
    sub_237: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_587, sum_214);  mul_587 = sum_214 = None
    sub_238: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_237, mul_589);  sub_237 = mul_589 = None
    mul_590: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_90, sub_238);  div_90 = sub_238 = None
    mul_591: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_245, mul_135);  mul_135 = None
    sum_216: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1]);  mul_591 = None
    sum_217: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_165: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_48, full_default_1, mul_590);  convert_element_type_48 = None
    mul_592: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_165, 1.1111111111111112);  where_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_869: "f32[512, 1536]" = torch.ops.aten.view.default(mul_592, [512, 1536]);  mul_592 = None
    mm_146: "f32[512, 6144]" = torch.ops.aten.mm.default(view_869, permute_738);  permute_738 = None
    permute_739: "f32[1536, 512]" = torch.ops.aten.permute.default(view_869, [1, 0])
    mm_147: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_739, view_262);  permute_739 = view_262 = None
    permute_740: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_218: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
    view_870: "f32[1536]" = torch.ops.aten.view.default(sum_218, [1536]);  sum_218 = None
    permute_741: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_740, [1, 0]);  permute_740 = None
    view_871: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_146, [1, 512, 6144]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_594: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
    mul_595: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_596: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_595, -0.5);  mul_595 = None
    exp_40: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_596);  mul_596 = None
    mul_597: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_598: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, mul_597);  view_261 = mul_597 = None
    add_247: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_594, mul_598);  mul_594 = mul_598 = None
    mul_599: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_871, add_247);  view_871 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_872: "f32[512, 6144]" = torch.ops.aten.view.default(mul_599, [512, 6144]);  mul_599 = None
    mm_148: "f32[512, 1536]" = torch.ops.aten.mm.default(view_872, permute_742);  permute_742 = None
    permute_743: "f32[6144, 512]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_149: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_743, view_260);  permute_743 = view_260 = None
    permute_744: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_219: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
    view_873: "f32[6144]" = torch.ops.aten.view.default(sum_219, [6144]);  sum_219 = None
    permute_745: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_744, [1, 0]);  permute_744 = None
    view_874: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_148, [1, 512, 1536]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_248: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_590, view_874);  mul_590 = view_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_601: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_248, primals_189);  primals_189 = None
    mul_602: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_601, 1536)
    sum_220: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True)
    mul_603: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_601, mul_129);  mul_601 = None
    sum_221: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [2], True);  mul_603 = None
    mul_604: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_129, sum_221);  sum_221 = None
    sub_240: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_602, sum_220);  mul_602 = sum_220 = None
    sub_241: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_240, mul_604);  sub_240 = mul_604 = None
    mul_605: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_91, sub_241);  div_91 = sub_241 = None
    mul_606: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_248, mul_129);  mul_129 = None
    sum_222: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 1]);  mul_606 = None
    sum_223: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_166: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_47, full_default_1, mul_605);  convert_element_type_47 = None
    mul_607: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_166, 1.1111111111111112);  where_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_875: "f32[512, 1536]" = torch.ops.aten.view.default(mul_607, [512, 1536]);  mul_607 = None
    mm_150: "f32[512, 1536]" = torch.ops.aten.mm.default(view_875, permute_746);  permute_746 = None
    permute_747: "f32[1536, 512]" = torch.ops.aten.permute.default(view_875, [1, 0])
    mm_151: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_747, view_258);  permute_747 = view_258 = None
    permute_748: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_224: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
    view_876: "f32[1536]" = torch.ops.aten.view.default(sum_224, [1536]);  sum_224 = None
    permute_749: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_877: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_150, [1, 512, 1536]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_878: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_877, [1, 512, 24, 64]);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_750: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_879: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_750, [24, 512, 64]);  permute_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_96: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_751, view_879);  permute_751 = None
    bmm_97: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_879, permute_752);  view_879 = permute_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_880: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_97, [1, 24, 512, 512]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_167: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, full_default_1, view_880);  convert_element_type_46 = view_880 = None
    mul_608: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_167, 1.1111111111111112);  where_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_609: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_608, alias_66);  mul_608 = None
    sum_225: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [-1], True)
    mul_610: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_66, sum_225);  alias_66 = sum_225 = None
    sub_242: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_881: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_242, [24, 512, 512]);  sub_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_98: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_753, view_881);  permute_753 = None
    bmm_99: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_881, permute_754);  view_881 = permute_754 = None
    div_92: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_98, full_default_2);  bmm_98 = None
    permute_755: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_92, [0, 2, 1]);  div_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_882: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_96, [1, 24, 512, 64]);  bmm_96 = None
    permute_756: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_122: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_756, memory_format = torch.contiguous_format);  permute_756 = None
    view_883: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_122, [1, 512, 1536]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_884: "f32[512, 1536]" = torch.ops.aten.view.default(view_883, [512, 1536]);  view_883 = None
    mm_152: "f32[512, 1536]" = torch.ops.aten.mm.default(view_884, permute_757);  permute_757 = None
    permute_758: "f32[1536, 512]" = torch.ops.aten.permute.default(view_884, [1, 0])
    mm_153: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_758, view_242);  permute_758 = None
    permute_759: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_226: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_884, [0], True);  view_884 = None
    view_885: "f32[1536]" = torch.ops.aten.view.default(sum_226, [1536]);  sum_226 = None
    permute_760: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_759, [1, 0]);  permute_759 = None
    view_886: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_152, [1, 512, 1536]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_249: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_605, view_886);  mul_605 = view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_887: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_755, [1, 24, 512, 64]);  permute_755 = None
    permute_761: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_887, [0, 2, 1, 3]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_888: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_761, [1, 512, 1536]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_889: "f32[512, 1536]" = torch.ops.aten.view.default(view_888, [512, 1536]);  view_888 = None
    mm_154: "f32[512, 1536]" = torch.ops.aten.mm.default(view_889, permute_762);  permute_762 = None
    permute_763: "f32[1536, 512]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_155: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_763, view_242);  permute_763 = None
    permute_764: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_227: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[1536]" = torch.ops.aten.view.default(sum_227, [1536]);  sum_227 = None
    permute_765: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_764, [1, 0]);  permute_764 = None
    view_891: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_154, [1, 512, 1536]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_250: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_249, view_891);  add_249 = view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_892: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_99, [1, 24, 512, 64]);  bmm_99 = None
    permute_766: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_123: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_766, memory_format = torch.contiguous_format);  permute_766 = None
    view_893: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_123, [1, 512, 1536]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_894: "f32[512, 1536]" = torch.ops.aten.view.default(view_893, [512, 1536]);  view_893 = None
    mm_156: "f32[512, 1536]" = torch.ops.aten.mm.default(view_894, permute_767);  permute_767 = None
    permute_768: "f32[1536, 512]" = torch.ops.aten.permute.default(view_894, [1, 0])
    mm_157: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_768, view_242);  permute_768 = view_242 = None
    permute_769: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_228: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
    view_895: "f32[1536]" = torch.ops.aten.view.default(sum_228, [1536]);  sum_228 = None
    permute_770: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_769, [1, 0]);  permute_769 = None
    view_896: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_156, [1, 512, 1536]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_251: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_250, view_896);  add_250 = view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_612: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_251, primals_179);  primals_179 = None
    mul_613: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_612, 1536)
    sum_229: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_612, [2], True)
    mul_614: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_612, mul_124);  mul_612 = None
    sum_230: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [2], True);  mul_614 = None
    mul_615: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_124, sum_230);  sum_230 = None
    sub_244: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_613, sum_229);  mul_613 = sum_229 = None
    sub_245: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_244, mul_615);  sub_244 = mul_615 = None
    mul_616: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_93, sub_245);  div_93 = sub_245 = None
    mul_617: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_251, mul_124);  mul_124 = None
    sum_231: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 1]);  mul_617 = None
    sum_232: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_168: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_44, full_default_1, mul_616);  convert_element_type_44 = None
    mul_618: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_168, 1.1111111111111112);  where_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_897: "f32[512, 1536]" = torch.ops.aten.view.default(mul_618, [512, 1536]);  mul_618 = None
    mm_158: "f32[512, 6144]" = torch.ops.aten.mm.default(view_897, permute_771);  permute_771 = None
    permute_772: "f32[1536, 512]" = torch.ops.aten.permute.default(view_897, [1, 0])
    mm_159: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_772, view_240);  permute_772 = view_240 = None
    permute_773: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_233: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True);  view_897 = None
    view_898: "f32[1536]" = torch.ops.aten.view.default(sum_233, [1536]);  sum_233 = None
    permute_774: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
    view_899: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_158, [1, 512, 6144]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_620: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
    mul_621: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_622: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_621, -0.5);  mul_621 = None
    exp_41: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_622);  mul_622 = None
    mul_623: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_624: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, mul_623);  view_239 = mul_623 = None
    add_253: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_620, mul_624);  mul_620 = mul_624 = None
    mul_625: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_899, add_253);  view_899 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_900: "f32[512, 6144]" = torch.ops.aten.view.default(mul_625, [512, 6144]);  mul_625 = None
    mm_160: "f32[512, 1536]" = torch.ops.aten.mm.default(view_900, permute_775);  permute_775 = None
    permute_776: "f32[6144, 512]" = torch.ops.aten.permute.default(view_900, [1, 0])
    mm_161: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_776, view_238);  permute_776 = view_238 = None
    permute_777: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_234: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True);  view_900 = None
    view_901: "f32[6144]" = torch.ops.aten.view.default(sum_234, [6144]);  sum_234 = None
    permute_778: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_902: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_160, [1, 512, 1536]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_254: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_616, view_902);  mul_616 = view_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_627: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_254, primals_173);  primals_173 = None
    mul_628: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_627, 1536)
    sum_235: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True)
    mul_629: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_627, mul_118);  mul_627 = None
    sum_236: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True);  mul_629 = None
    mul_630: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_118, sum_236);  sum_236 = None
    sub_247: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_628, sum_235);  mul_628 = sum_235 = None
    sub_248: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_247, mul_630);  sub_247 = mul_630 = None
    mul_631: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_94, sub_248);  div_94 = sub_248 = None
    mul_632: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_254, mul_118);  mul_118 = None
    sum_237: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 1]);  mul_632 = None
    sum_238: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_169: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_43, full_default_1, mul_631);  convert_element_type_43 = None
    mul_633: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_169, 1.1111111111111112);  where_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_903: "f32[512, 1536]" = torch.ops.aten.view.default(mul_633, [512, 1536]);  mul_633 = None
    mm_162: "f32[512, 1536]" = torch.ops.aten.mm.default(view_903, permute_779);  permute_779 = None
    permute_780: "f32[1536, 512]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_163: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_780, view_236);  permute_780 = view_236 = None
    permute_781: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_239: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[1536]" = torch.ops.aten.view.default(sum_239, [1536]);  sum_239 = None
    permute_782: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_905: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_162, [1, 512, 1536]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_906: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_905, [1, 512, 24, 64]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_783: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_907: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_783, [24, 512, 64]);  permute_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_100: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_784, view_907);  permute_784 = None
    bmm_101: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_907, permute_785);  view_907 = permute_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_908: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_101, [1, 24, 512, 512]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_170: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, full_default_1, view_908);  convert_element_type_42 = view_908 = None
    mul_634: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_170, 1.1111111111111112);  where_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_635: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_634, alias_69);  mul_634 = None
    sum_240: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [-1], True)
    mul_636: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_69, sum_240);  alias_69 = sum_240 = None
    sub_249: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_635, mul_636);  mul_635 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_909: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_249, [24, 512, 512]);  sub_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_102: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_786, view_909);  permute_786 = None
    bmm_103: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_909, permute_787);  view_909 = permute_787 = None
    div_95: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_102, full_default_2);  bmm_102 = None
    permute_788: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_95, [0, 2, 1]);  div_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_910: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_100, [1, 24, 512, 64]);  bmm_100 = None
    permute_789: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_910, [0, 2, 1, 3]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_124: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_789, memory_format = torch.contiguous_format);  permute_789 = None
    view_911: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_124, [1, 512, 1536]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_912: "f32[512, 1536]" = torch.ops.aten.view.default(view_911, [512, 1536]);  view_911 = None
    mm_164: "f32[512, 1536]" = torch.ops.aten.mm.default(view_912, permute_790);  permute_790 = None
    permute_791: "f32[1536, 512]" = torch.ops.aten.permute.default(view_912, [1, 0])
    mm_165: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_791, view_220);  permute_791 = None
    permute_792: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_241: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_912, [0], True);  view_912 = None
    view_913: "f32[1536]" = torch.ops.aten.view.default(sum_241, [1536]);  sum_241 = None
    permute_793: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_792, [1, 0]);  permute_792 = None
    view_914: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_164, [1, 512, 1536]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_255: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_631, view_914);  mul_631 = view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_915: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_788, [1, 24, 512, 64]);  permute_788 = None
    permute_794: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_915, [0, 2, 1, 3]);  view_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_916: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_794, [1, 512, 1536]);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_917: "f32[512, 1536]" = torch.ops.aten.view.default(view_916, [512, 1536]);  view_916 = None
    mm_166: "f32[512, 1536]" = torch.ops.aten.mm.default(view_917, permute_795);  permute_795 = None
    permute_796: "f32[1536, 512]" = torch.ops.aten.permute.default(view_917, [1, 0])
    mm_167: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_796, view_220);  permute_796 = None
    permute_797: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_242: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_917, [0], True);  view_917 = None
    view_918: "f32[1536]" = torch.ops.aten.view.default(sum_242, [1536]);  sum_242 = None
    permute_798: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    view_919: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_166, [1, 512, 1536]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_256: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_255, view_919);  add_255 = view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_920: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_103, [1, 24, 512, 64]);  bmm_103 = None
    permute_799: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_920, [0, 2, 1, 3]);  view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_125: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_799, memory_format = torch.contiguous_format);  permute_799 = None
    view_921: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_125, [1, 512, 1536]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_922: "f32[512, 1536]" = torch.ops.aten.view.default(view_921, [512, 1536]);  view_921 = None
    mm_168: "f32[512, 1536]" = torch.ops.aten.mm.default(view_922, permute_800);  permute_800 = None
    permute_801: "f32[1536, 512]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_169: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_801, view_220);  permute_801 = view_220 = None
    permute_802: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[1536]" = torch.ops.aten.view.default(sum_243, [1536]);  sum_243 = None
    permute_803: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    view_924: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_168, [1, 512, 1536]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_257: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_256, view_924);  add_256 = view_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_638: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_257, primals_163);  primals_163 = None
    mul_639: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_638, 1536)
    sum_244: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True)
    mul_640: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_638, mul_113);  mul_638 = None
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [2], True);  mul_640 = None
    mul_641: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_113, sum_245);  sum_245 = None
    sub_251: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_639, sum_244);  mul_639 = sum_244 = None
    sub_252: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_251, mul_641);  sub_251 = mul_641 = None
    mul_642: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_96, sub_252);  div_96 = sub_252 = None
    mul_643: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_257, mul_113);  mul_113 = None
    sum_246: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
    sum_247: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_171: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_40, full_default_1, mul_642);  convert_element_type_40 = None
    mul_644: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_171, 1.1111111111111112);  where_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_925: "f32[512, 1536]" = torch.ops.aten.view.default(mul_644, [512, 1536]);  mul_644 = None
    mm_170: "f32[512, 6144]" = torch.ops.aten.mm.default(view_925, permute_804);  permute_804 = None
    permute_805: "f32[1536, 512]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_171: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_805, view_218);  permute_805 = view_218 = None
    permute_806: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_248: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
    view_926: "f32[1536]" = torch.ops.aten.view.default(sum_248, [1536]);  sum_248 = None
    permute_807: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_806, [1, 0]);  permute_806 = None
    view_927: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_170, [1, 512, 6144]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_646: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_647: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_648: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_647, -0.5);  mul_647 = None
    exp_42: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_648);  mul_648 = None
    mul_649: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_650: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, mul_649);  view_217 = mul_649 = None
    add_259: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_646, mul_650);  mul_646 = mul_650 = None
    mul_651: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_927, add_259);  view_927 = add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_928: "f32[512, 6144]" = torch.ops.aten.view.default(mul_651, [512, 6144]);  mul_651 = None
    mm_172: "f32[512, 1536]" = torch.ops.aten.mm.default(view_928, permute_808);  permute_808 = None
    permute_809: "f32[6144, 512]" = torch.ops.aten.permute.default(view_928, [1, 0])
    mm_173: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_809, view_216);  permute_809 = view_216 = None
    permute_810: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_249: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_928, [0], True);  view_928 = None
    view_929: "f32[6144]" = torch.ops.aten.view.default(sum_249, [6144]);  sum_249 = None
    permute_811: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    view_930: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_172, [1, 512, 1536]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_260: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_642, view_930);  mul_642 = view_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_653: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_260, primals_157);  primals_157 = None
    mul_654: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_653, 1536)
    sum_250: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True)
    mul_655: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_653, mul_107);  mul_653 = None
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
    mul_656: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_107, sum_251);  sum_251 = None
    sub_254: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_654, sum_250);  mul_654 = sum_250 = None
    sub_255: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_254, mul_656);  sub_254 = mul_656 = None
    mul_657: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_97, sub_255);  div_97 = sub_255 = None
    mul_658: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_260, mul_107);  mul_107 = None
    sum_252: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 1]);  mul_658 = None
    sum_253: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_172: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_39, full_default_1, mul_657);  convert_element_type_39 = None
    mul_659: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_172, 1.1111111111111112);  where_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_931: "f32[512, 1536]" = torch.ops.aten.view.default(mul_659, [512, 1536]);  mul_659 = None
    mm_174: "f32[512, 1536]" = torch.ops.aten.mm.default(view_931, permute_812);  permute_812 = None
    permute_813: "f32[1536, 512]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_175: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_813, view_214);  permute_813 = view_214 = None
    permute_814: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_254: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[1536]" = torch.ops.aten.view.default(sum_254, [1536]);  sum_254 = None
    permute_815: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_933: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_174, [1, 512, 1536]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_934: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_933, [1, 512, 24, 64]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_816: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_935: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_816, [24, 512, 64]);  permute_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_104: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_817, view_935);  permute_817 = None
    bmm_105: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_935, permute_818);  view_935 = permute_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_936: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_105, [1, 24, 512, 512]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_173: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, full_default_1, view_936);  convert_element_type_38 = view_936 = None
    mul_660: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_173, 1.1111111111111112);  where_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_661: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_660, alias_72);  mul_660 = None
    sum_255: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [-1], True)
    mul_662: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_72, sum_255);  alias_72 = sum_255 = None
    sub_256: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_661, mul_662);  mul_661 = mul_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_937: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_256, [24, 512, 512]);  sub_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_106: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_819, view_937);  permute_819 = None
    bmm_107: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_937, permute_820);  view_937 = permute_820 = None
    div_98: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_106, full_default_2);  bmm_106 = None
    permute_821: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_98, [0, 2, 1]);  div_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_938: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_104, [1, 24, 512, 64]);  bmm_104 = None
    permute_822: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_938, [0, 2, 1, 3]);  view_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_126: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_822, memory_format = torch.contiguous_format);  permute_822 = None
    view_939: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_126, [1, 512, 1536]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_940: "f32[512, 1536]" = torch.ops.aten.view.default(view_939, [512, 1536]);  view_939 = None
    mm_176: "f32[512, 1536]" = torch.ops.aten.mm.default(view_940, permute_823);  permute_823 = None
    permute_824: "f32[1536, 512]" = torch.ops.aten.permute.default(view_940, [1, 0])
    mm_177: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_824, view_198);  permute_824 = None
    permute_825: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_256: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
    view_941: "f32[1536]" = torch.ops.aten.view.default(sum_256, [1536]);  sum_256 = None
    permute_826: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_825, [1, 0]);  permute_825 = None
    view_942: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_176, [1, 512, 1536]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_261: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_657, view_942);  mul_657 = view_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_943: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_821, [1, 24, 512, 64]);  permute_821 = None
    permute_827: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_943, [0, 2, 1, 3]);  view_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_944: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_827, [1, 512, 1536]);  permute_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_945: "f32[512, 1536]" = torch.ops.aten.view.default(view_944, [512, 1536]);  view_944 = None
    mm_178: "f32[512, 1536]" = torch.ops.aten.mm.default(view_945, permute_828);  permute_828 = None
    permute_829: "f32[1536, 512]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_179: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_829, view_198);  permute_829 = None
    permute_830: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_257: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[1536]" = torch.ops.aten.view.default(sum_257, [1536]);  sum_257 = None
    permute_831: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_830, [1, 0]);  permute_830 = None
    view_947: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_178, [1, 512, 1536]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_262: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_261, view_947);  add_261 = view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_948: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_107, [1, 24, 512, 64]);  bmm_107 = None
    permute_832: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_127: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_832, memory_format = torch.contiguous_format);  permute_832 = None
    view_949: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_127, [1, 512, 1536]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_950: "f32[512, 1536]" = torch.ops.aten.view.default(view_949, [512, 1536]);  view_949 = None
    mm_180: "f32[512, 1536]" = torch.ops.aten.mm.default(view_950, permute_833);  permute_833 = None
    permute_834: "f32[1536, 512]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_181: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_834, view_198);  permute_834 = view_198 = None
    permute_835: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_258: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
    view_951: "f32[1536]" = torch.ops.aten.view.default(sum_258, [1536]);  sum_258 = None
    permute_836: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_835, [1, 0]);  permute_835 = None
    view_952: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_180, [1, 512, 1536]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_263: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_262, view_952);  add_262 = view_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_664: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_263, primals_147);  primals_147 = None
    mul_665: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_664, 1536)
    sum_259: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_664, [2], True)
    mul_666: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_664, mul_102);  mul_664 = None
    sum_260: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True);  mul_666 = None
    mul_667: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_102, sum_260);  sum_260 = None
    sub_258: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_665, sum_259);  mul_665 = sum_259 = None
    sub_259: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_258, mul_667);  sub_258 = mul_667 = None
    mul_668: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_99, sub_259);  div_99 = sub_259 = None
    mul_669: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_263, mul_102);  mul_102 = None
    sum_261: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 1]);  mul_669 = None
    sum_262: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_174: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_36, full_default_1, mul_668);  convert_element_type_36 = None
    mul_670: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_174, 1.1111111111111112);  where_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_953: "f32[512, 1536]" = torch.ops.aten.view.default(mul_670, [512, 1536]);  mul_670 = None
    mm_182: "f32[512, 6144]" = torch.ops.aten.mm.default(view_953, permute_837);  permute_837 = None
    permute_838: "f32[1536, 512]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_183: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_838, view_196);  permute_838 = view_196 = None
    permute_839: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_263: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
    view_954: "f32[1536]" = torch.ops.aten.view.default(sum_263, [1536]);  sum_263 = None
    permute_840: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_839, [1, 0]);  permute_839 = None
    view_955: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_182, [1, 512, 6144]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_672: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_673: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_674: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_673, -0.5);  mul_673 = None
    exp_43: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_674);  mul_674 = None
    mul_675: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_676: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, mul_675);  view_195 = mul_675 = None
    add_265: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_672, mul_676);  mul_672 = mul_676 = None
    mul_677: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_955, add_265);  view_955 = add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_956: "f32[512, 6144]" = torch.ops.aten.view.default(mul_677, [512, 6144]);  mul_677 = None
    mm_184: "f32[512, 1536]" = torch.ops.aten.mm.default(view_956, permute_841);  permute_841 = None
    permute_842: "f32[6144, 512]" = torch.ops.aten.permute.default(view_956, [1, 0])
    mm_185: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_842, view_194);  permute_842 = view_194 = None
    permute_843: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_264: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_956, [0], True);  view_956 = None
    view_957: "f32[6144]" = torch.ops.aten.view.default(sum_264, [6144]);  sum_264 = None
    permute_844: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    view_958: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_184, [1, 512, 1536]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_266: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_668, view_958);  mul_668 = view_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_679: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_266, primals_141);  primals_141 = None
    mul_680: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_679, 1536)
    sum_265: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True)
    mul_681: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_679, mul_96);  mul_679 = None
    sum_266: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_681, [2], True);  mul_681 = None
    mul_682: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_96, sum_266);  sum_266 = None
    sub_261: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_680, sum_265);  mul_680 = sum_265 = None
    sub_262: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_261, mul_682);  sub_261 = mul_682 = None
    mul_683: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_100, sub_262);  div_100 = sub_262 = None
    mul_684: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_266, mul_96);  mul_96 = None
    sum_267: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 1]);  mul_684 = None
    sum_268: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_175: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_35, full_default_1, mul_683);  convert_element_type_35 = None
    mul_685: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_175, 1.1111111111111112);  where_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_959: "f32[512, 1536]" = torch.ops.aten.view.default(mul_685, [512, 1536]);  mul_685 = None
    mm_186: "f32[512, 1536]" = torch.ops.aten.mm.default(view_959, permute_845);  permute_845 = None
    permute_846: "f32[1536, 512]" = torch.ops.aten.permute.default(view_959, [1, 0])
    mm_187: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_846, view_192);  permute_846 = view_192 = None
    permute_847: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_269: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_959, [0], True);  view_959 = None
    view_960: "f32[1536]" = torch.ops.aten.view.default(sum_269, [1536]);  sum_269 = None
    permute_848: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    view_961: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_186, [1, 512, 1536]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_962: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_961, [1, 512, 24, 64]);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_849: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_962, [0, 2, 1, 3]);  view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_963: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_849, [24, 512, 64]);  permute_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_108: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_850, view_963);  permute_850 = None
    bmm_109: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_963, permute_851);  view_963 = permute_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_964: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_109, [1, 24, 512, 512]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_176: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, full_default_1, view_964);  convert_element_type_34 = view_964 = None
    mul_686: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_176, 1.1111111111111112);  where_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_687: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_686, alias_75);  mul_686 = None
    sum_270: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [-1], True)
    mul_688: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_75, sum_270);  alias_75 = sum_270 = None
    sub_263: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_965: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_263, [24, 512, 512]);  sub_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_110: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_852, view_965);  permute_852 = None
    bmm_111: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_965, permute_853);  view_965 = permute_853 = None
    div_101: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_110, full_default_2);  bmm_110 = None
    permute_854: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_101, [0, 2, 1]);  div_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_966: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_108, [1, 24, 512, 64]);  bmm_108 = None
    permute_855: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_966, [0, 2, 1, 3]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_128: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_855, memory_format = torch.contiguous_format);  permute_855 = None
    view_967: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_128, [1, 512, 1536]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_968: "f32[512, 1536]" = torch.ops.aten.view.default(view_967, [512, 1536]);  view_967 = None
    mm_188: "f32[512, 1536]" = torch.ops.aten.mm.default(view_968, permute_856);  permute_856 = None
    permute_857: "f32[1536, 512]" = torch.ops.aten.permute.default(view_968, [1, 0])
    mm_189: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_857, view_176);  permute_857 = None
    permute_858: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_271: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_968, [0], True);  view_968 = None
    view_969: "f32[1536]" = torch.ops.aten.view.default(sum_271, [1536]);  sum_271 = None
    permute_859: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_858, [1, 0]);  permute_858 = None
    view_970: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_188, [1, 512, 1536]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_267: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_683, view_970);  mul_683 = view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_971: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_854, [1, 24, 512, 64]);  permute_854 = None
    permute_860: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_971, [0, 2, 1, 3]);  view_971 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_972: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_860, [1, 512, 1536]);  permute_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_973: "f32[512, 1536]" = torch.ops.aten.view.default(view_972, [512, 1536]);  view_972 = None
    mm_190: "f32[512, 1536]" = torch.ops.aten.mm.default(view_973, permute_861);  permute_861 = None
    permute_862: "f32[1536, 512]" = torch.ops.aten.permute.default(view_973, [1, 0])
    mm_191: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_862, view_176);  permute_862 = None
    permute_863: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_272: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_973, [0], True);  view_973 = None
    view_974: "f32[1536]" = torch.ops.aten.view.default(sum_272, [1536]);  sum_272 = None
    permute_864: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
    view_975: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_190, [1, 512, 1536]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_268: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_267, view_975);  add_267 = view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_976: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_111, [1, 24, 512, 64]);  bmm_111 = None
    permute_865: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_129: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_865, memory_format = torch.contiguous_format);  permute_865 = None
    view_977: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_129, [1, 512, 1536]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_978: "f32[512, 1536]" = torch.ops.aten.view.default(view_977, [512, 1536]);  view_977 = None
    mm_192: "f32[512, 1536]" = torch.ops.aten.mm.default(view_978, permute_866);  permute_866 = None
    permute_867: "f32[1536, 512]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_193: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_867, view_176);  permute_867 = view_176 = None
    permute_868: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_273: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
    view_979: "f32[1536]" = torch.ops.aten.view.default(sum_273, [1536]);  sum_273 = None
    permute_869: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_868, [1, 0]);  permute_868 = None
    view_980: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_192, [1, 512, 1536]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_269: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_268, view_980);  add_268 = view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_690: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_269, primals_131);  primals_131 = None
    mul_691: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_690, 1536)
    sum_274: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_690, [2], True)
    mul_692: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_690, mul_91);  mul_690 = None
    sum_275: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_692, [2], True);  mul_692 = None
    mul_693: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_91, sum_275);  sum_275 = None
    sub_265: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_691, sum_274);  mul_691 = sum_274 = None
    sub_266: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_265, mul_693);  sub_265 = mul_693 = None
    mul_694: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_102, sub_266);  div_102 = sub_266 = None
    mul_695: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_269, mul_91);  mul_91 = None
    sum_276: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 1]);  mul_695 = None
    sum_277: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_177: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_32, full_default_1, mul_694);  convert_element_type_32 = None
    mul_696: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_177, 1.1111111111111112);  where_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_981: "f32[512, 1536]" = torch.ops.aten.view.default(mul_696, [512, 1536]);  mul_696 = None
    mm_194: "f32[512, 6144]" = torch.ops.aten.mm.default(view_981, permute_870);  permute_870 = None
    permute_871: "f32[1536, 512]" = torch.ops.aten.permute.default(view_981, [1, 0])
    mm_195: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_871, view_174);  permute_871 = view_174 = None
    permute_872: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_278: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_981, [0], True);  view_981 = None
    view_982: "f32[1536]" = torch.ops.aten.view.default(sum_278, [1536]);  sum_278 = None
    permute_873: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_872, [1, 0]);  permute_872 = None
    view_983: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_194, [1, 512, 6144]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_698: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_699: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_700: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_699, -0.5);  mul_699 = None
    exp_44: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_700);  mul_700 = None
    mul_701: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_702: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, mul_701);  view_173 = mul_701 = None
    add_271: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_698, mul_702);  mul_698 = mul_702 = None
    mul_703: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_983, add_271);  view_983 = add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_984: "f32[512, 6144]" = torch.ops.aten.view.default(mul_703, [512, 6144]);  mul_703 = None
    mm_196: "f32[512, 1536]" = torch.ops.aten.mm.default(view_984, permute_874);  permute_874 = None
    permute_875: "f32[6144, 512]" = torch.ops.aten.permute.default(view_984, [1, 0])
    mm_197: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_875, view_172);  permute_875 = view_172 = None
    permute_876: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_279: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_984, [0], True);  view_984 = None
    view_985: "f32[6144]" = torch.ops.aten.view.default(sum_279, [6144]);  sum_279 = None
    permute_877: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    view_986: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_196, [1, 512, 1536]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_272: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_694, view_986);  mul_694 = view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_705: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_272, primals_125);  primals_125 = None
    mul_706: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_705, 1536)
    sum_280: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
    mul_707: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_705, mul_85);  mul_705 = None
    sum_281: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
    mul_708: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_85, sum_281);  sum_281 = None
    sub_268: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_706, sum_280);  mul_706 = sum_280 = None
    sub_269: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_268, mul_708);  sub_268 = mul_708 = None
    mul_709: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_103, sub_269);  div_103 = sub_269 = None
    mul_710: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_272, mul_85);  mul_85 = None
    sum_282: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
    sum_283: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_178: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_31, full_default_1, mul_709);  convert_element_type_31 = None
    mul_711: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_178, 1.1111111111111112);  where_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_987: "f32[512, 1536]" = torch.ops.aten.view.default(mul_711, [512, 1536]);  mul_711 = None
    mm_198: "f32[512, 1536]" = torch.ops.aten.mm.default(view_987, permute_878);  permute_878 = None
    permute_879: "f32[1536, 512]" = torch.ops.aten.permute.default(view_987, [1, 0])
    mm_199: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_879, view_170);  permute_879 = view_170 = None
    permute_880: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_284: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_987, [0], True);  view_987 = None
    view_988: "f32[1536]" = torch.ops.aten.view.default(sum_284, [1536]);  sum_284 = None
    permute_881: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_880, [1, 0]);  permute_880 = None
    view_989: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_198, [1, 512, 1536]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_990: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_989, [1, 512, 24, 64]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_882: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_990, [0, 2, 1, 3]);  view_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_991: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_882, [24, 512, 64]);  permute_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_112: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_883, view_991);  permute_883 = None
    bmm_113: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_991, permute_884);  view_991 = permute_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_992: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_113, [1, 24, 512, 512]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_179: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, full_default_1, view_992);  convert_element_type_30 = view_992 = None
    mul_712: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_179, 1.1111111111111112);  where_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_713: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_712, alias_78);  mul_712 = None
    sum_285: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [-1], True)
    mul_714: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_78, sum_285);  alias_78 = sum_285 = None
    sub_270: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_713, mul_714);  mul_713 = mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_993: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_270, [24, 512, 512]);  sub_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_114: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_885, view_993);  permute_885 = None
    bmm_115: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_993, permute_886);  view_993 = permute_886 = None
    div_104: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_114, full_default_2);  bmm_114 = None
    permute_887: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_104, [0, 2, 1]);  div_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_994: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_112, [1, 24, 512, 64]);  bmm_112 = None
    permute_888: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_994, [0, 2, 1, 3]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_130: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_888, memory_format = torch.contiguous_format);  permute_888 = None
    view_995: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_130, [1, 512, 1536]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_996: "f32[512, 1536]" = torch.ops.aten.view.default(view_995, [512, 1536]);  view_995 = None
    mm_200: "f32[512, 1536]" = torch.ops.aten.mm.default(view_996, permute_889);  permute_889 = None
    permute_890: "f32[1536, 512]" = torch.ops.aten.permute.default(view_996, [1, 0])
    mm_201: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_890, view_154);  permute_890 = None
    permute_891: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_286: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_996, [0], True);  view_996 = None
    view_997: "f32[1536]" = torch.ops.aten.view.default(sum_286, [1536]);  sum_286 = None
    permute_892: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_891, [1, 0]);  permute_891 = None
    view_998: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_200, [1, 512, 1536]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_273: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_709, view_998);  mul_709 = view_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_999: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_887, [1, 24, 512, 64]);  permute_887 = None
    permute_893: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_999, [0, 2, 1, 3]);  view_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1000: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_893, [1, 512, 1536]);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1001: "f32[512, 1536]" = torch.ops.aten.view.default(view_1000, [512, 1536]);  view_1000 = None
    mm_202: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1001, permute_894);  permute_894 = None
    permute_895: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1001, [1, 0])
    mm_203: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_895, view_154);  permute_895 = None
    permute_896: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_287: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1001, [0], True);  view_1001 = None
    view_1002: "f32[1536]" = torch.ops.aten.view.default(sum_287, [1536]);  sum_287 = None
    permute_897: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_896, [1, 0]);  permute_896 = None
    view_1003: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_202, [1, 512, 1536]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_274: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_273, view_1003);  add_273 = view_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1004: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_115, [1, 24, 512, 64]);  bmm_115 = None
    permute_898: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1004, [0, 2, 1, 3]);  view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_131: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_898, memory_format = torch.contiguous_format);  permute_898 = None
    view_1005: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_131, [1, 512, 1536]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1006: "f32[512, 1536]" = torch.ops.aten.view.default(view_1005, [512, 1536]);  view_1005 = None
    mm_204: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1006, permute_899);  permute_899 = None
    permute_900: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1006, [1, 0])
    mm_205: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_900, view_154);  permute_900 = view_154 = None
    permute_901: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_288: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True);  view_1006 = None
    view_1007: "f32[1536]" = torch.ops.aten.view.default(sum_288, [1536]);  sum_288 = None
    permute_902: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_901, [1, 0]);  permute_901 = None
    view_1008: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_204, [1, 512, 1536]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_275: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_274, view_1008);  add_274 = view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_716: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_275, primals_115);  primals_115 = None
    mul_717: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_716, 1536)
    sum_289: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True)
    mul_718: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_716, mul_80);  mul_716 = None
    sum_290: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True);  mul_718 = None
    mul_719: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_80, sum_290);  sum_290 = None
    sub_272: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_717, sum_289);  mul_717 = sum_289 = None
    sub_273: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_272, mul_719);  sub_272 = mul_719 = None
    mul_720: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_105, sub_273);  div_105 = sub_273 = None
    mul_721: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_275, mul_80);  mul_80 = None
    sum_291: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1]);  mul_721 = None
    sum_292: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_180: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_28, full_default_1, mul_720);  convert_element_type_28 = None
    mul_722: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_180, 1.1111111111111112);  where_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1009: "f32[512, 1536]" = torch.ops.aten.view.default(mul_722, [512, 1536]);  mul_722 = None
    mm_206: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1009, permute_903);  permute_903 = None
    permute_904: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_207: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_904, view_152);  permute_904 = view_152 = None
    permute_905: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_293: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
    view_1010: "f32[1536]" = torch.ops.aten.view.default(sum_293, [1536]);  sum_293 = None
    permute_906: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_905, [1, 0]);  permute_905 = None
    view_1011: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_206, [1, 512, 6144]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_724: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_725: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_726: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_725, -0.5);  mul_725 = None
    exp_45: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_726);  mul_726 = None
    mul_727: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_728: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, mul_727);  view_151 = mul_727 = None
    add_277: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_724, mul_728);  mul_724 = mul_728 = None
    mul_729: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1011, add_277);  view_1011 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1012: "f32[512, 6144]" = torch.ops.aten.view.default(mul_729, [512, 6144]);  mul_729 = None
    mm_208: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1012, permute_907);  permute_907 = None
    permute_908: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1012, [1, 0])
    mm_209: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_908, view_150);  permute_908 = view_150 = None
    permute_909: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_294: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
    view_1013: "f32[6144]" = torch.ops.aten.view.default(sum_294, [6144]);  sum_294 = None
    permute_910: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_909, [1, 0]);  permute_909 = None
    view_1014: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_208, [1, 512, 1536]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_278: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_720, view_1014);  mul_720 = view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_731: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_278, primals_109);  primals_109 = None
    mul_732: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_731, 1536)
    sum_295: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
    mul_733: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_731, mul_74);  mul_731 = None
    sum_296: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
    mul_734: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_74, sum_296);  sum_296 = None
    sub_275: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_732, sum_295);  mul_732 = sum_295 = None
    sub_276: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_275, mul_734);  sub_275 = mul_734 = None
    mul_735: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_106, sub_276);  div_106 = sub_276 = None
    mul_736: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_278, mul_74);  mul_74 = None
    sum_297: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
    sum_298: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_181: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_27, full_default_1, mul_735);  convert_element_type_27 = None
    mul_737: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_181, 1.1111111111111112);  where_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1015: "f32[512, 1536]" = torch.ops.aten.view.default(mul_737, [512, 1536]);  mul_737 = None
    mm_210: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1015, permute_911);  permute_911 = None
    permute_912: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1015, [1, 0])
    mm_211: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_912, view_148);  permute_912 = view_148 = None
    permute_913: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_299: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
    view_1016: "f32[1536]" = torch.ops.aten.view.default(sum_299, [1536]);  sum_299 = None
    permute_914: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
    view_1017: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_210, [1, 512, 1536]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1018: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1017, [1, 512, 24, 64]);  view_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_915: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1018, [0, 2, 1, 3]);  view_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1019: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_915, [24, 512, 64]);  permute_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_116: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_916, view_1019);  permute_916 = None
    bmm_117: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1019, permute_917);  view_1019 = permute_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1020: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_117, [1, 24, 512, 512]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_182: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, full_default_1, view_1020);  convert_element_type_26 = view_1020 = None
    mul_738: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_182, 1.1111111111111112);  where_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_739: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_738, alias_81);  mul_738 = None
    sum_300: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_739, [-1], True)
    mul_740: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_81, sum_300);  alias_81 = sum_300 = None
    sub_277: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1021: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_277, [24, 512, 512]);  sub_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_118: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_918, view_1021);  permute_918 = None
    bmm_119: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1021, permute_919);  view_1021 = permute_919 = None
    div_107: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_118, full_default_2);  bmm_118 = None
    permute_920: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_107, [0, 2, 1]);  div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1022: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_116, [1, 24, 512, 64]);  bmm_116 = None
    permute_921: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1022, [0, 2, 1, 3]);  view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_132: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_921, memory_format = torch.contiguous_format);  permute_921 = None
    view_1023: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_132, [1, 512, 1536]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1024: "f32[512, 1536]" = torch.ops.aten.view.default(view_1023, [512, 1536]);  view_1023 = None
    mm_212: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1024, permute_922);  permute_922 = None
    permute_923: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1024, [1, 0])
    mm_213: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_923, view_132);  permute_923 = None
    permute_924: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_301: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1024, [0], True);  view_1024 = None
    view_1025: "f32[1536]" = torch.ops.aten.view.default(sum_301, [1536]);  sum_301 = None
    permute_925: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_924, [1, 0]);  permute_924 = None
    view_1026: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_212, [1, 512, 1536]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_279: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_735, view_1026);  mul_735 = view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1027: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_920, [1, 24, 512, 64]);  permute_920 = None
    permute_926: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1027, [0, 2, 1, 3]);  view_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1028: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_926, [1, 512, 1536]);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1029: "f32[512, 1536]" = torch.ops.aten.view.default(view_1028, [512, 1536]);  view_1028 = None
    mm_214: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1029, permute_927);  permute_927 = None
    permute_928: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1029, [1, 0])
    mm_215: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_928, view_132);  permute_928 = None
    permute_929: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_302: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1029, [0], True);  view_1029 = None
    view_1030: "f32[1536]" = torch.ops.aten.view.default(sum_302, [1536]);  sum_302 = None
    permute_930: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_929, [1, 0]);  permute_929 = None
    view_1031: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_214, [1, 512, 1536]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_280: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_279, view_1031);  add_279 = view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1032: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_119, [1, 24, 512, 64]);  bmm_119 = None
    permute_931: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1032, [0, 2, 1, 3]);  view_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_133: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_931, memory_format = torch.contiguous_format);  permute_931 = None
    view_1033: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_133, [1, 512, 1536]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1034: "f32[512, 1536]" = torch.ops.aten.view.default(view_1033, [512, 1536]);  view_1033 = None
    mm_216: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1034, permute_932);  permute_932 = None
    permute_933: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_217: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_933, view_132);  permute_933 = view_132 = None
    permute_934: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_303: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
    view_1035: "f32[1536]" = torch.ops.aten.view.default(sum_303, [1536]);  sum_303 = None
    permute_935: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_934, [1, 0]);  permute_934 = None
    view_1036: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_216, [1, 512, 1536]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_281: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_280, view_1036);  add_280 = view_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_742: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_281, primals_99);  primals_99 = None
    mul_743: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_742, 1536)
    sum_304: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [2], True)
    mul_744: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_742, mul_69);  mul_742 = None
    sum_305: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True);  mul_744 = None
    mul_745: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_69, sum_305);  sum_305 = None
    sub_279: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_743, sum_304);  mul_743 = sum_304 = None
    sub_280: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_279, mul_745);  sub_279 = mul_745 = None
    mul_746: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_108, sub_280);  div_108 = sub_280 = None
    mul_747: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_281, mul_69);  mul_69 = None
    sum_306: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 1]);  mul_747 = None
    sum_307: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_183: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_24, full_default_1, mul_746);  convert_element_type_24 = None
    mul_748: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_183, 1.1111111111111112);  where_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1037: "f32[512, 1536]" = torch.ops.aten.view.default(mul_748, [512, 1536]);  mul_748 = None
    mm_218: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1037, permute_936);  permute_936 = None
    permute_937: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1037, [1, 0])
    mm_219: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_937, view_130);  permute_937 = view_130 = None
    permute_938: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_308: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
    view_1038: "f32[1536]" = torch.ops.aten.view.default(sum_308, [1536]);  sum_308 = None
    permute_939: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_938, [1, 0]);  permute_938 = None
    view_1039: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_218, [1, 512, 6144]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_750: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_751: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_752: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_751, -0.5);  mul_751 = None
    exp_46: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_752);  mul_752 = None
    mul_753: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_754: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, mul_753);  view_129 = mul_753 = None
    add_283: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_750, mul_754);  mul_750 = mul_754 = None
    mul_755: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1039, add_283);  view_1039 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1040: "f32[512, 6144]" = torch.ops.aten.view.default(mul_755, [512, 6144]);  mul_755 = None
    mm_220: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1040, permute_940);  permute_940 = None
    permute_941: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1040, [1, 0])
    mm_221: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_941, view_128);  permute_941 = view_128 = None
    permute_942: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_309: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True);  view_1040 = None
    view_1041: "f32[6144]" = torch.ops.aten.view.default(sum_309, [6144]);  sum_309 = None
    permute_943: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_942, [1, 0]);  permute_942 = None
    view_1042: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_220, [1, 512, 1536]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_284: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_746, view_1042);  mul_746 = view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_757: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_284, primals_93);  primals_93 = None
    mul_758: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_757, 1536)
    sum_310: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [2], True)
    mul_759: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_757, mul_63);  mul_757 = None
    sum_311: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_759, [2], True);  mul_759 = None
    mul_760: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_63, sum_311);  sum_311 = None
    sub_282: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_758, sum_310);  mul_758 = sum_310 = None
    sub_283: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_282, mul_760);  sub_282 = mul_760 = None
    mul_761: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_109, sub_283);  div_109 = sub_283 = None
    mul_762: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_284, mul_63);  mul_63 = None
    sum_312: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 1]);  mul_762 = None
    sum_313: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_184: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_23, full_default_1, mul_761);  convert_element_type_23 = None
    mul_763: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_184, 1.1111111111111112);  where_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1043: "f32[512, 1536]" = torch.ops.aten.view.default(mul_763, [512, 1536]);  mul_763 = None
    mm_222: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1043, permute_944);  permute_944 = None
    permute_945: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_223: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_945, view_126);  permute_945 = view_126 = None
    permute_946: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_314: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[1536]" = torch.ops.aten.view.default(sum_314, [1536]);  sum_314 = None
    permute_947: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_946, [1, 0]);  permute_946 = None
    view_1045: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_222, [1, 512, 1536]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1046: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1045, [1, 512, 24, 64]);  view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_948: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1047: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_948, [24, 512, 64]);  permute_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_120: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_949, view_1047);  permute_949 = None
    bmm_121: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1047, permute_950);  view_1047 = permute_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1048: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_121, [1, 24, 512, 512]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_185: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, full_default_1, view_1048);  convert_element_type_22 = view_1048 = None
    mul_764: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_185, 1.1111111111111112);  where_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_765: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_764, alias_84);  mul_764 = None
    sum_315: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [-1], True)
    mul_766: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_84, sum_315);  alias_84 = sum_315 = None
    sub_284: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_765, mul_766);  mul_765 = mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1049: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_284, [24, 512, 512]);  sub_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_122: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_951, view_1049);  permute_951 = None
    bmm_123: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1049, permute_952);  view_1049 = permute_952 = None
    div_110: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_122, full_default_2);  bmm_122 = None
    permute_953: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_110, [0, 2, 1]);  div_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1050: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_120, [1, 24, 512, 64]);  bmm_120 = None
    permute_954: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1050, [0, 2, 1, 3]);  view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_134: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_954, memory_format = torch.contiguous_format);  permute_954 = None
    view_1051: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_134, [1, 512, 1536]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1052: "f32[512, 1536]" = torch.ops.aten.view.default(view_1051, [512, 1536]);  view_1051 = None
    mm_224: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1052, permute_955);  permute_955 = None
    permute_956: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1052, [1, 0])
    mm_225: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_956, view_110);  permute_956 = None
    permute_957: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_316: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1052, [0], True);  view_1052 = None
    view_1053: "f32[1536]" = torch.ops.aten.view.default(sum_316, [1536]);  sum_316 = None
    permute_958: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_957, [1, 0]);  permute_957 = None
    view_1054: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_224, [1, 512, 1536]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_285: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_761, view_1054);  mul_761 = view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1055: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_953, [1, 24, 512, 64]);  permute_953 = None
    permute_959: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1055, [0, 2, 1, 3]);  view_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1056: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_959, [1, 512, 1536]);  permute_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1057: "f32[512, 1536]" = torch.ops.aten.view.default(view_1056, [512, 1536]);  view_1056 = None
    mm_226: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1057, permute_960);  permute_960 = None
    permute_961: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1057, [1, 0])
    mm_227: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_961, view_110);  permute_961 = None
    permute_962: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_317: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1057, [0], True);  view_1057 = None
    view_1058: "f32[1536]" = torch.ops.aten.view.default(sum_317, [1536]);  sum_317 = None
    permute_963: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_962, [1, 0]);  permute_962 = None
    view_1059: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_226, [1, 512, 1536]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_286: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_285, view_1059);  add_285 = view_1059 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1060: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_123, [1, 24, 512, 64]);  bmm_123 = None
    permute_964: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1060, [0, 2, 1, 3]);  view_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_135: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
    view_1061: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_135, [1, 512, 1536]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1062: "f32[512, 1536]" = torch.ops.aten.view.default(view_1061, [512, 1536]);  view_1061 = None
    mm_228: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1062, permute_965);  permute_965 = None
    permute_966: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_229: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_966, view_110);  permute_966 = view_110 = None
    permute_967: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_318: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
    view_1063: "f32[1536]" = torch.ops.aten.view.default(sum_318, [1536]);  sum_318 = None
    permute_968: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_967, [1, 0]);  permute_967 = None
    view_1064: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_228, [1, 512, 1536]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_287: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_286, view_1064);  add_286 = view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_768: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_287, primals_83);  primals_83 = None
    mul_769: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_768, 1536)
    sum_319: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True)
    mul_770: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_768, mul_58);  mul_768 = None
    sum_320: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True);  mul_770 = None
    mul_771: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_58, sum_320);  sum_320 = None
    sub_286: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_769, sum_319);  mul_769 = sum_319 = None
    sub_287: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_286, mul_771);  sub_286 = mul_771 = None
    mul_772: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_111, sub_287);  div_111 = sub_287 = None
    mul_773: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_287, mul_58);  mul_58 = None
    sum_321: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 1]);  mul_773 = None
    sum_322: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_186: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_20, full_default_1, mul_772);  convert_element_type_20 = None
    mul_774: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_186, 1.1111111111111112);  where_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1065: "f32[512, 1536]" = torch.ops.aten.view.default(mul_774, [512, 1536]);  mul_774 = None
    mm_230: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1065, permute_969);  permute_969 = None
    permute_970: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1065, [1, 0])
    mm_231: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_970, view_108);  permute_970 = view_108 = None
    permute_971: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_323: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
    view_1066: "f32[1536]" = torch.ops.aten.view.default(sum_323, [1536]);  sum_323 = None
    permute_972: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_971, [1, 0]);  permute_971 = None
    view_1067: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_230, [1, 512, 6144]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_776: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_777: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_778: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_777, -0.5);  mul_777 = None
    exp_47: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_778);  mul_778 = None
    mul_779: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_780: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, mul_779);  view_107 = mul_779 = None
    add_289: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_776, mul_780);  mul_776 = mul_780 = None
    mul_781: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1067, add_289);  view_1067 = add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1068: "f32[512, 6144]" = torch.ops.aten.view.default(mul_781, [512, 6144]);  mul_781 = None
    mm_232: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1068, permute_973);  permute_973 = None
    permute_974: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1068, [1, 0])
    mm_233: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_974, view_106);  permute_974 = view_106 = None
    permute_975: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_324: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True);  view_1068 = None
    view_1069: "f32[6144]" = torch.ops.aten.view.default(sum_324, [6144]);  sum_324 = None
    permute_976: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_975, [1, 0]);  permute_975 = None
    view_1070: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_232, [1, 512, 1536]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_290: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_772, view_1070);  mul_772 = view_1070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_783: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_290, primals_77);  primals_77 = None
    mul_784: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_783, 1536)
    sum_325: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_783, [2], True)
    mul_785: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_783, mul_52);  mul_783 = None
    sum_326: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [2], True);  mul_785 = None
    mul_786: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_52, sum_326);  sum_326 = None
    sub_289: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_784, sum_325);  mul_784 = sum_325 = None
    sub_290: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_289, mul_786);  sub_289 = mul_786 = None
    mul_787: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_112, sub_290);  div_112 = sub_290 = None
    mul_788: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_290, mul_52);  mul_52 = None
    sum_327: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 1]);  mul_788 = None
    sum_328: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_187: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_19, full_default_1, mul_787);  convert_element_type_19 = None
    mul_789: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_187, 1.1111111111111112);  where_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1071: "f32[512, 1536]" = torch.ops.aten.view.default(mul_789, [512, 1536]);  mul_789 = None
    mm_234: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1071, permute_977);  permute_977 = None
    permute_978: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1071, [1, 0])
    mm_235: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_978, view_104);  permute_978 = view_104 = None
    permute_979: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_329: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1071, [0], True);  view_1071 = None
    view_1072: "f32[1536]" = torch.ops.aten.view.default(sum_329, [1536]);  sum_329 = None
    permute_980: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_979, [1, 0]);  permute_979 = None
    view_1073: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_234, [1, 512, 1536]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1074: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1073, [1, 512, 24, 64]);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_981: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1074, [0, 2, 1, 3]);  view_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1075: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_981, [24, 512, 64]);  permute_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_124: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_982, view_1075);  permute_982 = None
    bmm_125: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1075, permute_983);  view_1075 = permute_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1076: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_125, [1, 24, 512, 512]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_188: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, full_default_1, view_1076);  convert_element_type_18 = view_1076 = None
    mul_790: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_188, 1.1111111111111112);  where_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_791: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_790, alias_87);  mul_790 = None
    sum_330: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [-1], True)
    mul_792: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_87, sum_330);  alias_87 = sum_330 = None
    sub_291: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1077: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_291, [24, 512, 512]);  sub_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_126: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_984, view_1077);  permute_984 = None
    bmm_127: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1077, permute_985);  view_1077 = permute_985 = None
    div_113: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_126, full_default_2);  bmm_126 = None
    permute_986: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_113, [0, 2, 1]);  div_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1078: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_124, [1, 24, 512, 64]);  bmm_124 = None
    permute_987: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1078, [0, 2, 1, 3]);  view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_136: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_987, memory_format = torch.contiguous_format);  permute_987 = None
    view_1079: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_136, [1, 512, 1536]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1080: "f32[512, 1536]" = torch.ops.aten.view.default(view_1079, [512, 1536]);  view_1079 = None
    mm_236: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1080, permute_988);  permute_988 = None
    permute_989: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1080, [1, 0])
    mm_237: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_989, view_88);  permute_989 = None
    permute_990: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_331: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
    view_1081: "f32[1536]" = torch.ops.aten.view.default(sum_331, [1536]);  sum_331 = None
    permute_991: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_990, [1, 0]);  permute_990 = None
    view_1082: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_236, [1, 512, 1536]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_291: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_787, view_1082);  mul_787 = view_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1083: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_986, [1, 24, 512, 64]);  permute_986 = None
    permute_992: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1083, [0, 2, 1, 3]);  view_1083 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1084: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_992, [1, 512, 1536]);  permute_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1085: "f32[512, 1536]" = torch.ops.aten.view.default(view_1084, [512, 1536]);  view_1084 = None
    mm_238: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1085, permute_993);  permute_993 = None
    permute_994: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1085, [1, 0])
    mm_239: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_994, view_88);  permute_994 = None
    permute_995: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_332: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1085, [0], True);  view_1085 = None
    view_1086: "f32[1536]" = torch.ops.aten.view.default(sum_332, [1536]);  sum_332 = None
    permute_996: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_995, [1, 0]);  permute_995 = None
    view_1087: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_238, [1, 512, 1536]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_292: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_291, view_1087);  add_291 = view_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1088: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_127, [1, 24, 512, 64]);  bmm_127 = None
    permute_997: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1088, [0, 2, 1, 3]);  view_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_137: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_997, memory_format = torch.contiguous_format);  permute_997 = None
    view_1089: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_137, [1, 512, 1536]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1090: "f32[512, 1536]" = torch.ops.aten.view.default(view_1089, [512, 1536]);  view_1089 = None
    mm_240: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1090, permute_998);  permute_998 = None
    permute_999: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1090, [1, 0])
    mm_241: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_999, view_88);  permute_999 = view_88 = None
    permute_1000: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_333: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
    view_1091: "f32[1536]" = torch.ops.aten.view.default(sum_333, [1536]);  sum_333 = None
    permute_1001: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1000, [1, 0]);  permute_1000 = None
    view_1092: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_240, [1, 512, 1536]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_293: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_292, view_1092);  add_292 = view_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_794: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_293, primals_67);  primals_67 = None
    mul_795: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_794, 1536)
    sum_334: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_794, [2], True)
    mul_796: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_794, mul_47);  mul_794 = None
    sum_335: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_796, [2], True);  mul_796 = None
    mul_797: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_47, sum_335);  sum_335 = None
    sub_293: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_795, sum_334);  mul_795 = sum_334 = None
    sub_294: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_293, mul_797);  sub_293 = mul_797 = None
    mul_798: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_114, sub_294);  div_114 = sub_294 = None
    mul_799: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_293, mul_47);  mul_47 = None
    sum_336: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 1]);  mul_799 = None
    sum_337: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_189: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_16, full_default_1, mul_798);  convert_element_type_16 = None
    mul_800: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_189, 1.1111111111111112);  where_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1093: "f32[512, 1536]" = torch.ops.aten.view.default(mul_800, [512, 1536]);  mul_800 = None
    mm_242: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1093, permute_1002);  permute_1002 = None
    permute_1003: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1093, [1, 0])
    mm_243: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1003, view_86);  permute_1003 = view_86 = None
    permute_1004: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_338: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
    view_1094: "f32[1536]" = torch.ops.aten.view.default(sum_338, [1536]);  sum_338 = None
    permute_1005: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_1004, [1, 0]);  permute_1004 = None
    view_1095: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_242, [1, 512, 6144]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_802: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_803: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_804: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_803, -0.5);  mul_803 = None
    exp_48: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_804);  mul_804 = None
    mul_805: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_806: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, mul_805);  view_85 = mul_805 = None
    add_295: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_802, mul_806);  mul_802 = mul_806 = None
    mul_807: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1095, add_295);  view_1095 = add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1096: "f32[512, 6144]" = torch.ops.aten.view.default(mul_807, [512, 6144]);  mul_807 = None
    mm_244: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1096, permute_1006);  permute_1006 = None
    permute_1007: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1096, [1, 0])
    mm_245: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1007, view_84);  permute_1007 = view_84 = None
    permute_1008: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_339: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True);  view_1096 = None
    view_1097: "f32[6144]" = torch.ops.aten.view.default(sum_339, [6144]);  sum_339 = None
    permute_1009: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_1008, [1, 0]);  permute_1008 = None
    view_1098: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_244, [1, 512, 1536]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_296: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_798, view_1098);  mul_798 = view_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_809: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_296, primals_61);  primals_61 = None
    mul_810: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_809, 1536)
    sum_340: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_809, [2], True)
    mul_811: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_809, mul_41);  mul_809 = None
    sum_341: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [2], True);  mul_811 = None
    mul_812: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_41, sum_341);  sum_341 = None
    sub_296: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_810, sum_340);  mul_810 = sum_340 = None
    sub_297: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_296, mul_812);  sub_296 = mul_812 = None
    mul_813: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_115, sub_297);  div_115 = sub_297 = None
    mul_814: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_296, mul_41);  mul_41 = None
    sum_342: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 1]);  mul_814 = None
    sum_343: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_190: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_15, full_default_1, mul_813);  convert_element_type_15 = None
    mul_815: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_190, 1.1111111111111112);  where_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1099: "f32[512, 1536]" = torch.ops.aten.view.default(mul_815, [512, 1536]);  mul_815 = None
    mm_246: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1099, permute_1010);  permute_1010 = None
    permute_1011: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_247: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1011, view_82);  permute_1011 = view_82 = None
    permute_1012: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_344: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
    view_1100: "f32[1536]" = torch.ops.aten.view.default(sum_344, [1536]);  sum_344 = None
    permute_1013: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1012, [1, 0]);  permute_1012 = None
    view_1101: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_246, [1, 512, 1536]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1102: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1101, [1, 512, 24, 64]);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_1014: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1102, [0, 2, 1, 3]);  view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1103: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_1014, [24, 512, 64]);  permute_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_128: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_1015, view_1103);  permute_1015 = None
    bmm_129: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1103, permute_1016);  view_1103 = permute_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1104: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_129, [1, 24, 512, 512]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_191: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, full_default_1, view_1104);  convert_element_type_14 = view_1104 = None
    mul_816: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_191, 1.1111111111111112);  where_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_817: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_816, alias_90);  mul_816 = None
    sum_345: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [-1], True)
    mul_818: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_90, sum_345);  alias_90 = sum_345 = None
    sub_298: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1105: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_298, [24, 512, 512]);  sub_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_130: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_1017, view_1105);  permute_1017 = None
    bmm_131: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1105, permute_1018);  view_1105 = permute_1018 = None
    div_116: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_130, full_default_2);  bmm_130 = None
    permute_1019: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_116, [0, 2, 1]);  div_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1106: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_128, [1, 24, 512, 64]);  bmm_128 = None
    permute_1020: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1106, [0, 2, 1, 3]);  view_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_138: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1020, memory_format = torch.contiguous_format);  permute_1020 = None
    view_1107: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_138, [1, 512, 1536]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1108: "f32[512, 1536]" = torch.ops.aten.view.default(view_1107, [512, 1536]);  view_1107 = None
    mm_248: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1108, permute_1021);  permute_1021 = None
    permute_1022: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1108, [1, 0])
    mm_249: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1022, view_66);  permute_1022 = None
    permute_1023: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_346: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1108, [0], True);  view_1108 = None
    view_1109: "f32[1536]" = torch.ops.aten.view.default(sum_346, [1536]);  sum_346 = None
    permute_1024: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1023, [1, 0]);  permute_1023 = None
    view_1110: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_248, [1, 512, 1536]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_297: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_813, view_1110);  mul_813 = view_1110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1111: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_1019, [1, 24, 512, 64]);  permute_1019 = None
    permute_1025: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1111, [0, 2, 1, 3]);  view_1111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1112: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_1025, [1, 512, 1536]);  permute_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1113: "f32[512, 1536]" = torch.ops.aten.view.default(view_1112, [512, 1536]);  view_1112 = None
    mm_250: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1113, permute_1026);  permute_1026 = None
    permute_1027: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1113, [1, 0])
    mm_251: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1027, view_66);  permute_1027 = None
    permute_1028: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_347: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1113, [0], True);  view_1113 = None
    view_1114: "f32[1536]" = torch.ops.aten.view.default(sum_347, [1536]);  sum_347 = None
    permute_1029: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
    view_1115: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_250, [1, 512, 1536]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_298: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_297, view_1115);  add_297 = view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1116: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_131, [1, 24, 512, 64]);  bmm_131 = None
    permute_1030: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1116, [0, 2, 1, 3]);  view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_139: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1030, memory_format = torch.contiguous_format);  permute_1030 = None
    view_1117: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_139, [1, 512, 1536]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1118: "f32[512, 1536]" = torch.ops.aten.view.default(view_1117, [512, 1536]);  view_1117 = None
    mm_252: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1118, permute_1031);  permute_1031 = None
    permute_1032: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1118, [1, 0])
    mm_253: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1032, view_66);  permute_1032 = view_66 = None
    permute_1033: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_348: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
    view_1119: "f32[1536]" = torch.ops.aten.view.default(sum_348, [1536]);  sum_348 = None
    permute_1034: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1033, [1, 0]);  permute_1033 = None
    view_1120: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_252, [1, 512, 1536]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_299: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_298, view_1120);  add_298 = view_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_820: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_299, primals_51);  primals_51 = None
    mul_821: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_820, 1536)
    sum_349: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True)
    mul_822: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_820, mul_36);  mul_820 = None
    sum_350: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_822, [2], True);  mul_822 = None
    mul_823: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_36, sum_350);  sum_350 = None
    sub_300: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_821, sum_349);  mul_821 = sum_349 = None
    sub_301: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_300, mul_823);  sub_300 = mul_823 = None
    mul_824: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_117, sub_301);  div_117 = sub_301 = None
    mul_825: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_299, mul_36);  mul_36 = None
    sum_351: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 1]);  mul_825 = None
    sum_352: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_192: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_12, full_default_1, mul_824);  convert_element_type_12 = None
    mul_826: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_192, 1.1111111111111112);  where_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1121: "f32[512, 1536]" = torch.ops.aten.view.default(mul_826, [512, 1536]);  mul_826 = None
    mm_254: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1121, permute_1035);  permute_1035 = None
    permute_1036: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1121, [1, 0])
    mm_255: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1036, view_64);  permute_1036 = view_64 = None
    permute_1037: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_353: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
    view_1122: "f32[1536]" = torch.ops.aten.view.default(sum_353, [1536]);  sum_353 = None
    permute_1038: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_1037, [1, 0]);  permute_1037 = None
    view_1123: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_254, [1, 512, 6144]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_828: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_829: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_830: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_829, -0.5);  mul_829 = None
    exp_49: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_830);  mul_830 = None
    mul_831: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_832: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, mul_831);  view_63 = mul_831 = None
    add_301: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_828, mul_832);  mul_828 = mul_832 = None
    mul_833: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1123, add_301);  view_1123 = add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1124: "f32[512, 6144]" = torch.ops.aten.view.default(mul_833, [512, 6144]);  mul_833 = None
    mm_256: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1124, permute_1039);  permute_1039 = None
    permute_1040: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1124, [1, 0])
    mm_257: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1040, view_62);  permute_1040 = view_62 = None
    permute_1041: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_354: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1124, [0], True);  view_1124 = None
    view_1125: "f32[6144]" = torch.ops.aten.view.default(sum_354, [6144]);  sum_354 = None
    permute_1042: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_1041, [1, 0]);  permute_1041 = None
    view_1126: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_256, [1, 512, 1536]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_302: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_824, view_1126);  mul_824 = view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_835: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_302, primals_45);  primals_45 = None
    mul_836: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_835, 1536)
    sum_355: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True)
    mul_837: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_835, mul_30);  mul_835 = None
    sum_356: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2], True);  mul_837 = None
    mul_838: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_30, sum_356);  sum_356 = None
    sub_303: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_836, sum_355);  mul_836 = sum_355 = None
    sub_304: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_303, mul_838);  sub_303 = mul_838 = None
    mul_839: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_118, sub_304);  div_118 = sub_304 = None
    mul_840: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_302, mul_30);  mul_30 = None
    sum_357: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 1]);  mul_840 = None
    sum_358: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_193: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_11, full_default_1, mul_839);  convert_element_type_11 = None
    mul_841: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_193, 1.1111111111111112);  where_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1127: "f32[512, 1536]" = torch.ops.aten.view.default(mul_841, [512, 1536]);  mul_841 = None
    mm_258: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1127, permute_1043);  permute_1043 = None
    permute_1044: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1127, [1, 0])
    mm_259: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1044, view_60);  permute_1044 = view_60 = None
    permute_1045: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_359: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1127, [0], True);  view_1127 = None
    view_1128: "f32[1536]" = torch.ops.aten.view.default(sum_359, [1536]);  sum_359 = None
    permute_1046: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1045, [1, 0]);  permute_1045 = None
    view_1129: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_258, [1, 512, 1536]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1130: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1129, [1, 512, 24, 64]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_1047: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1130, [0, 2, 1, 3]);  view_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1131: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_1047, [24, 512, 64]);  permute_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_132: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_1048, view_1131);  permute_1048 = None
    bmm_133: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1131, permute_1049);  view_1131 = permute_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1132: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_133, [1, 24, 512, 512]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_194: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, full_default_1, view_1132);  convert_element_type_10 = view_1132 = None
    mul_842: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_194, 1.1111111111111112);  where_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_843: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_842, alias_93);  mul_842 = None
    sum_360: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [-1], True)
    mul_844: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_93, sum_360);  alias_93 = sum_360 = None
    sub_305: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1133: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_305, [24, 512, 512]);  sub_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_134: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_1050, view_1133);  permute_1050 = None
    bmm_135: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1133, permute_1051);  view_1133 = permute_1051 = None
    div_119: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_134, full_default_2);  bmm_134 = None
    permute_1052: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_119, [0, 2, 1]);  div_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1134: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_132, [1, 24, 512, 64]);  bmm_132 = None
    permute_1053: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1134, [0, 2, 1, 3]);  view_1134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_140: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1053, memory_format = torch.contiguous_format);  permute_1053 = None
    view_1135: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_140, [1, 512, 1536]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1136: "f32[512, 1536]" = torch.ops.aten.view.default(view_1135, [512, 1536]);  view_1135 = None
    mm_260: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1136, permute_1054);  permute_1054 = None
    permute_1055: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1136, [1, 0])
    mm_261: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1055, view_44);  permute_1055 = None
    permute_1056: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_361: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1136, [0], True);  view_1136 = None
    view_1137: "f32[1536]" = torch.ops.aten.view.default(sum_361, [1536]);  sum_361 = None
    permute_1057: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1056, [1, 0]);  permute_1056 = None
    view_1138: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_260, [1, 512, 1536]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_303: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_839, view_1138);  mul_839 = view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1139: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_1052, [1, 24, 512, 64]);  permute_1052 = None
    permute_1058: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1, 3]);  view_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1140: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_1058, [1, 512, 1536]);  permute_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1141: "f32[512, 1536]" = torch.ops.aten.view.default(view_1140, [512, 1536]);  view_1140 = None
    mm_262: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1141, permute_1059);  permute_1059 = None
    permute_1060: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1141, [1, 0])
    mm_263: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1060, view_44);  permute_1060 = None
    permute_1061: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_362: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1141, [0], True);  view_1141 = None
    view_1142: "f32[1536]" = torch.ops.aten.view.default(sum_362, [1536]);  sum_362 = None
    permute_1062: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1061, [1, 0]);  permute_1061 = None
    view_1143: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_262, [1, 512, 1536]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_304: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_303, view_1143);  add_303 = view_1143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1144: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_135, [1, 24, 512, 64]);  bmm_135 = None
    permute_1063: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1144, [0, 2, 1, 3]);  view_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_141: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1063, memory_format = torch.contiguous_format);  permute_1063 = None
    view_1145: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_141, [1, 512, 1536]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1146: "f32[512, 1536]" = torch.ops.aten.view.default(view_1145, [512, 1536]);  view_1145 = None
    mm_264: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1146, permute_1064);  permute_1064 = None
    permute_1065: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1146, [1, 0])
    mm_265: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1065, view_44);  permute_1065 = view_44 = None
    permute_1066: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_363: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True);  view_1146 = None
    view_1147: "f32[1536]" = torch.ops.aten.view.default(sum_363, [1536]);  sum_363 = None
    permute_1067: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1066, [1, 0]);  permute_1066 = None
    view_1148: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_264, [1, 512, 1536]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_305: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_304, view_1148);  add_304 = view_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_846: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_305, primals_35);  primals_35 = None
    mul_847: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_846, 1536)
    sum_364: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_846, [2], True)
    mul_848: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_846, mul_25);  mul_846 = None
    sum_365: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_848, [2], True);  mul_848 = None
    mul_849: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_25, sum_365);  sum_365 = None
    sub_307: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_847, sum_364);  mul_847 = sum_364 = None
    sub_308: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_307, mul_849);  sub_307 = mul_849 = None
    mul_850: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_120, sub_308);  div_120 = sub_308 = None
    mul_851: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_305, mul_25);  mul_25 = None
    sum_366: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 1]);  mul_851 = None
    sum_367: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_195: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_8, full_default_1, mul_850);  convert_element_type_8 = None
    mul_852: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_195, 1.1111111111111112);  where_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1149: "f32[512, 1536]" = torch.ops.aten.view.default(mul_852, [512, 1536]);  mul_852 = None
    mm_266: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1149, permute_1068);  permute_1068 = None
    permute_1069: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1149, [1, 0])
    mm_267: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1069, view_42);  permute_1069 = view_42 = None
    permute_1070: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_368: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1149, [0], True);  view_1149 = None
    view_1150: "f32[1536]" = torch.ops.aten.view.default(sum_368, [1536]);  sum_368 = None
    permute_1071: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_1070, [1, 0]);  permute_1070 = None
    view_1151: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_266, [1, 512, 6144]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_854: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_855: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_856: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_855, -0.5);  mul_855 = None
    exp_50: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_856);  mul_856 = None
    mul_857: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_858: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, mul_857);  view_41 = mul_857 = None
    add_307: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_854, mul_858);  mul_854 = mul_858 = None
    mul_859: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1151, add_307);  view_1151 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1152: "f32[512, 6144]" = torch.ops.aten.view.default(mul_859, [512, 6144]);  mul_859 = None
    mm_268: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1152, permute_1072);  permute_1072 = None
    permute_1073: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1152, [1, 0])
    mm_269: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1073, view_40);  permute_1073 = view_40 = None
    permute_1074: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_369: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1152, [0], True);  view_1152 = None
    view_1153: "f32[6144]" = torch.ops.aten.view.default(sum_369, [6144]);  sum_369 = None
    permute_1075: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_1074, [1, 0]);  permute_1074 = None
    view_1154: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_268, [1, 512, 1536]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_308: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_850, view_1154);  mul_850 = view_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_861: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_308, primals_29);  primals_29 = None
    mul_862: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_861, 1536)
    sum_370: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True)
    mul_863: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_861, mul_19);  mul_861 = None
    sum_371: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_863, [2], True);  mul_863 = None
    mul_864: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_19, sum_371);  sum_371 = None
    sub_310: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_862, sum_370);  mul_862 = sum_370 = None
    sub_311: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_310, mul_864);  sub_310 = mul_864 = None
    mul_865: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_121, sub_311);  div_121 = sub_311 = None
    mul_866: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_308, mul_19);  mul_19 = None
    sum_372: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_866, [0, 1]);  mul_866 = None
    sum_373: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_196: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_7, full_default_1, mul_865);  convert_element_type_7 = None
    mul_867: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_196, 1.1111111111111112);  where_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1155: "f32[512, 1536]" = torch.ops.aten.view.default(mul_867, [512, 1536]);  mul_867 = None
    mm_270: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1155, permute_1076);  permute_1076 = None
    permute_1077: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1155, [1, 0])
    mm_271: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1077, view_38);  permute_1077 = view_38 = None
    permute_1078: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_374: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1155, [0], True);  view_1155 = None
    view_1156: "f32[1536]" = torch.ops.aten.view.default(sum_374, [1536]);  sum_374 = None
    permute_1079: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1078, [1, 0]);  permute_1078 = None
    view_1157: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_270, [1, 512, 1536]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1158: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1157, [1, 512, 24, 64]);  view_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_1080: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1158, [0, 2, 1, 3]);  view_1158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1159: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_1080, [24, 512, 64]);  permute_1080 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_136: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_1081, view_1159);  permute_1081 = None
    bmm_137: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1159, permute_1082);  view_1159 = permute_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1160: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_137, [1, 24, 512, 512]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_197: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, full_default_1, view_1160);  convert_element_type_6 = view_1160 = None
    mul_868: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_197, 1.1111111111111112);  where_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_869: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_868, alias_96);  mul_868 = None
    sum_375: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_869, [-1], True)
    mul_870: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_96, sum_375);  alias_96 = sum_375 = None
    sub_312: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1161: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_312, [24, 512, 512]);  sub_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_138: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_1083, view_1161);  permute_1083 = None
    bmm_139: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1161, permute_1084);  view_1161 = permute_1084 = None
    div_122: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_138, full_default_2);  bmm_138 = None
    permute_1085: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_122, [0, 2, 1]);  div_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1162: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_136, [1, 24, 512, 64]);  bmm_136 = None
    permute_1086: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1162, [0, 2, 1, 3]);  view_1162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_142: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1086, memory_format = torch.contiguous_format);  permute_1086 = None
    view_1163: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_142, [1, 512, 1536]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1164: "f32[512, 1536]" = torch.ops.aten.view.default(view_1163, [512, 1536]);  view_1163 = None
    mm_272: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1164, permute_1087);  permute_1087 = None
    permute_1088: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1164, [1, 0])
    mm_273: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1088, view_22);  permute_1088 = None
    permute_1089: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_376: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1164, [0], True);  view_1164 = None
    view_1165: "f32[1536]" = torch.ops.aten.view.default(sum_376, [1536]);  sum_376 = None
    permute_1090: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1089, [1, 0]);  permute_1089 = None
    view_1166: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_272, [1, 512, 1536]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_309: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_865, view_1166);  mul_865 = view_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1167: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_1085, [1, 24, 512, 64]);  permute_1085 = None
    permute_1091: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1167, [0, 2, 1, 3]);  view_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1168: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_1091, [1, 512, 1536]);  permute_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1169: "f32[512, 1536]" = torch.ops.aten.view.default(view_1168, [512, 1536]);  view_1168 = None
    mm_274: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1169, permute_1092);  permute_1092 = None
    permute_1093: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1169, [1, 0])
    mm_275: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1093, view_22);  permute_1093 = None
    permute_1094: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_377: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1169, [0], True);  view_1169 = None
    view_1170: "f32[1536]" = torch.ops.aten.view.default(sum_377, [1536]);  sum_377 = None
    permute_1095: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1094, [1, 0]);  permute_1094 = None
    view_1171: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_274, [1, 512, 1536]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_310: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_309, view_1171);  add_309 = view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1172: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_139, [1, 24, 512, 64]);  bmm_139 = None
    permute_1096: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1172, [0, 2, 1, 3]);  view_1172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_143: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1096, memory_format = torch.contiguous_format);  permute_1096 = None
    view_1173: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_143, [1, 512, 1536]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1174: "f32[512, 1536]" = torch.ops.aten.view.default(view_1173, [512, 1536]);  view_1173 = None
    mm_276: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1174, permute_1097);  permute_1097 = None
    permute_1098: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1174, [1, 0])
    mm_277: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1098, view_22);  permute_1098 = view_22 = None
    permute_1099: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_378: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
    view_1175: "f32[1536]" = torch.ops.aten.view.default(sum_378, [1536]);  sum_378 = None
    permute_1100: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1099, [1, 0]);  permute_1099 = None
    view_1176: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_276, [1, 512, 1536]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_311: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_310, view_1176);  add_310 = view_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:341, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_872: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_311, primals_19);  primals_19 = None
    mul_873: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_872, 1536)
    sum_379: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [2], True)
    mul_874: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_872, mul_14);  mul_872 = None
    sum_380: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_874, [2], True);  mul_874 = None
    mul_875: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_14, sum_380);  sum_380 = None
    sub_314: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_873, sum_379);  mul_873 = sum_379 = None
    sub_315: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_314, mul_875);  sub_314 = mul_875 = None
    mul_876: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_123, sub_315);  div_123 = sub_315 = None
    mul_877: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_311, mul_14);  mul_14 = None
    sum_381: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 1]);  mul_877 = None
    sum_382: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_198: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_4, full_default_1, mul_876);  convert_element_type_4 = None
    mul_878: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_198, 1.1111111111111112);  where_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:339, code: hidden_states = self.dense(hidden_states)
    view_1177: "f32[512, 1536]" = torch.ops.aten.view.default(mul_878, [512, 1536]);  mul_878 = None
    mm_278: "f32[512, 6144]" = torch.ops.aten.mm.default(view_1177, permute_1101);  permute_1101 = None
    permute_1102: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1177, [1, 0])
    mm_279: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1102, view_20);  permute_1102 = view_20 = None
    permute_1103: "f32[6144, 1536]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_383: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1177, [0], True);  view_1177 = None
    view_1178: "f32[1536]" = torch.ops.aten.view.default(sum_383, [1536]);  sum_383 = None
    permute_1104: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_1103, [1, 0]);  permute_1103 = None
    view_1179: "f32[1, 512, 6144]" = torch.ops.aten.view.default(mm_278, [1, 512, 6144]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_880: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_881: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_882: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_881, -0.5);  mul_881 = None
    exp_51: "f32[1, 512, 6144]" = torch.ops.aten.exp.default(mul_882);  mul_882 = None
    mul_883: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_884: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, mul_883);  view_19 = mul_883 = None
    add_313: "f32[1, 512, 6144]" = torch.ops.aten.add.Tensor(mul_880, mul_884);  mul_880 = mul_884 = None
    mul_885: "f32[1, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1179, add_313);  view_1179 = add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    view_1180: "f32[512, 6144]" = torch.ops.aten.view.default(mul_885, [512, 6144]);  mul_885 = None
    mm_280: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1180, permute_1105);  permute_1105 = None
    permute_1106: "f32[6144, 512]" = torch.ops.aten.permute.default(view_1180, [1, 0])
    mm_281: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1106, view_18);  permute_1106 = view_18 = None
    permute_1107: "f32[1536, 6144]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_384: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1180, [0], True);  view_1180 = None
    view_1181: "f32[6144]" = torch.ops.aten.view.default(sum_384, [6144]);  sum_384 = None
    permute_1108: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_1107, [1, 0]);  permute_1107 = None
    view_1182: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_280, [1, 512, 1536]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:324, code: hidden_states = self.dense(hidden_states)
    add_314: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_876, view_1182);  mul_876 = view_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:272, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_887: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_314, primals_13);  primals_13 = None
    mul_888: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_887, 1536)
    sum_385: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
    mul_889: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_887, mul_8);  mul_887 = None
    sum_386: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
    mul_890: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_8, sum_386);  sum_386 = None
    sub_317: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_888, sum_385);  mul_888 = sum_385 = None
    sub_318: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_317, mul_890);  sub_317 = mul_890 = None
    mul_891: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_124, sub_318);  div_124 = sub_318 = None
    mul_892: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(add_314, mul_8);  mul_8 = None
    sum_387: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
    sum_388: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_199: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type_3, full_default_1, mul_891);  convert_element_type_3 = None
    mul_893: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_199, 1.1111111111111112);  where_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:270, code: hidden_states = self.dense(hidden_states)
    view_1183: "f32[512, 1536]" = torch.ops.aten.view.default(mul_893, [512, 1536]);  mul_893 = None
    mm_282: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1183, permute_1109);  permute_1109 = None
    permute_1110: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1183, [1, 0])
    mm_283: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1110, view_16);  permute_1110 = view_16 = None
    permute_1111: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_389: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
    view_1184: "f32[1536]" = torch.ops.aten.view.default(sum_389, [1536]);  sum_389 = None
    permute_1112: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1111, [1, 0]);  permute_1111 = None
    view_1185: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_282, [1, 512, 1536]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:750, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1186: "f32[1, 512, 24, 64]" = torch.ops.aten.view.default(view_1185, [1, 512, 24, 64]);  view_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:746, code: .permute(0, 2, 1, 3)
    permute_1113: "f32[1, 24, 512, 64]" = torch.ops.aten.permute.default(view_1186, [0, 2, 1, 3]);  view_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:745, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
    view_1187: "f32[24, 512, 64]" = torch.ops.aten.view.default(permute_1113, [24, 512, 64]);  permute_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:741, code: context_layer = torch.bmm(
    bmm_140: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(permute_1114, view_1187);  permute_1114 = None
    bmm_141: "f32[24, 512, 512]" = torch.ops.aten.bmm.default(view_1187, permute_1115);  view_1187 = permute_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:742, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
    view_1188: "f32[1, 24, 512, 512]" = torch.ops.aten.view.default(bmm_141, [1, 24, 512, 512]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_200: "f32[1, 24, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, full_default_1, view_1188);  convert_element_type_2 = view_1188 = None
    mul_894: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(where_200, 1.1111111111111112);  where_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:739, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_895: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_894, alias_99);  mul_894 = None
    sum_390: "f32[1, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_895, [-1], True)
    mul_896: "f32[1, 24, 512, 512]" = torch.ops.aten.mul.Tensor(alias_99, sum_390);  alias_99 = sum_390 = None
    sub_319: "f32[1, 24, 512, 512]" = torch.ops.aten.sub.Tensor(mul_895, mul_896);  mul_895 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:734, code: attention_scores = attention_scores.view(
    view_1189: "f32[24, 512, 512]" = torch.ops.aten.view.default(sub_319, [24, 512, 512]);  sub_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:724, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
    bmm_142: "f32[24, 64, 512]" = torch.ops.aten.bmm.default(permute_1116, view_1189);  permute_1116 = None
    bmm_143: "f32[24, 512, 64]" = torch.ops.aten.bmm.default(view_1189, permute_1117);  view_1189 = permute_1117 = None
    div_125: "f32[24, 64, 512]" = torch.ops.aten.div.Tensor(bmm_142, full_default_2);  bmm_142 = full_default_2 = None
    permute_1118: "f32[24, 512, 64]" = torch.ops.aten.permute.default(div_125, [0, 2, 1]);  div_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1190: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_140, [1, 24, 512, 64]);  bmm_140 = None
    permute_1119: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1190, [0, 2, 1, 3]);  view_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_144: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1119, memory_format = torch.contiguous_format);  permute_1119 = None
    view_1191: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_144, [1, 512, 1536]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    view_1192: "f32[512, 1536]" = torch.ops.aten.view.default(view_1191, [512, 1536]);  view_1191 = None
    mm_284: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1192, permute_1120);  permute_1120 = None
    permute_1121: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1192, [1, 0])
    mm_285: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1121, view);  permute_1121 = None
    permute_1122: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_391: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1192, [0], True);  view_1192 = None
    view_1193: "f32[1536]" = torch.ops.aten.view.default(sum_391, [1536]);  sum_391 = None
    permute_1123: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1122, [1, 0]);  permute_1122 = None
    view_1194: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_284, [1, 512, 1536]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:714, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    add_315: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(mul_891, view_1194);  mul_891 = view_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1195: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(permute_1118, [1, 24, 512, 64]);  permute_1118 = None
    permute_1124: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1195, [0, 2, 1, 3]);  view_1195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    view_1196: "f32[1, 512, 1536]" = torch.ops.aten.view.default(permute_1124, [1, 512, 1536]);  permute_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    view_1197: "f32[512, 1536]" = torch.ops.aten.view.default(view_1196, [512, 1536]);  view_1196 = None
    mm_286: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1197, permute_1125);  permute_1125 = None
    permute_1126: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1197, [1, 0])
    mm_287: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1126, view);  permute_1126 = None
    permute_1127: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    sum_392: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1197, [0], True);  view_1197 = None
    view_1198: "f32[1536]" = torch.ops.aten.view.default(sum_392, [1536]);  sum_392 = None
    permute_1128: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1127, [1, 0]);  permute_1127 = None
    view_1199: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_286, [1, 512, 1536]);  mm_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:713, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    add_316: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_315, view_1199);  add_315 = view_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:670, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    view_1200: "f32[1, 24, 512, 64]" = torch.ops.aten.view.default(bmm_143, [1, 24, 512, 64]);  bmm_143 = None
    permute_1129: "f32[1, 512, 24, 64]" = torch.ops.aten.permute.default(view_1200, [0, 2, 1, 3]);  view_1200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:669, code: x = x.view(new_x_shape)
    clone_145: "f32[1, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1129, memory_format = torch.contiguous_format);  permute_1129 = None
    view_1201: "f32[1, 512, 1536]" = torch.ops.aten.view.default(clone_145, [1, 512, 1536]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    view_1202: "f32[512, 1536]" = torch.ops.aten.view.default(view_1201, [512, 1536]);  view_1201 = None
    mm_288: "f32[512, 1536]" = torch.ops.aten.mm.default(view_1202, permute_1130);  permute_1130 = None
    permute_1131: "f32[1536, 512]" = torch.ops.aten.permute.default(view_1202, [1, 0])
    mm_289: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1131, view);  permute_1131 = view = None
    permute_1132: "f32[1536, 1536]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    sum_393: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True);  view_1202 = None
    view_1203: "f32[1536]" = torch.ops.aten.view.default(sum_393, [1536]);  sum_393 = None
    permute_1133: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_1132, [1, 0]);  permute_1132 = None
    view_1204: "f32[1, 512, 1536]" = torch.ops.aten.view.default(mm_288, [1, 512, 1536]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:712, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    add_317: "f32[1, 512, 1536]" = torch.ops.aten.add.Tensor(add_316, view_1204);  add_316 = view_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:234, code: return XDropout.apply(x, self.get_context())
    where_201: "f32[1, 512, 1536]" = torch.ops.aten.where.self(convert_element_type, full_default_1, add_317);  convert_element_type = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:910, code: embeddings = embeddings * mask
    mul_897: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(where_201, 1.1111111111111112);  where_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:901, code: embeddings = self.LayerNorm(embeddings)
    mul_900: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_897, primals_3);  primals_3 = None
    mul_901: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_900, 1536)
    sum_394: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_900, [2], True)
    mul_902: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_900, mul);  mul_900 = None
    sum_395: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_902, [2], True);  mul_902 = None
    mul_903: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, sum_395);  sum_395 = None
    sub_321: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_901, sum_394);  mul_901 = sum_394 = None
    sub_322: "f32[1, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_321, mul_903);  sub_321 = mul_903 = None
    mul_904: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(div_126, sub_322);  div_126 = sub_322 = None
    mul_905: "f32[1, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_897, mul);  mul = None
    sum_396: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_905, [0, 1]);  mul_905 = None
    sum_397: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 1]);  mul_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:887, code: position_embeddings = self.position_embeddings(position_ids.long())
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_202: "f32[1, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_10, full_default_1, mul_904);  unsqueeze_10 = None
    full_default_254: "f32[512, 1536]" = torch.ops.aten.full.default([512, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 1536]" = torch.ops.aten._unsafe_index_put.default(full_default_254, [slice_1], where_202, True);  full_default_254 = slice_1 = where_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:884, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_392, 0)
    unsqueeze_11: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_203: "f32[1, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_11, full_default_1, mul_904);  unsqueeze_11 = full_default_1 = mul_904 = None
    full_default_256: "f32[128100, 1536]" = torch.ops.aten.full.default([128100, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[128100, 1536]" = torch.ops.aten._unsafe_index_put.default(full_default_256, [primals_392], where_203, True);  full_default_256 = primals_392 = where_203 = None
    return [_unsafe_index_put_1, _unsafe_index_put, sum_396, sum_397, permute_1133, view_1203, permute_1128, view_1198, permute_1123, view_1193, permute_1112, view_1184, sum_387, sum_388, permute_1108, view_1181, permute_1104, view_1178, sum_381, sum_382, permute_1100, view_1175, permute_1095, view_1170, permute_1090, view_1165, permute_1079, view_1156, sum_372, sum_373, permute_1075, view_1153, permute_1071, view_1150, sum_366, sum_367, permute_1067, view_1147, permute_1062, view_1142, permute_1057, view_1137, permute_1046, view_1128, sum_357, sum_358, permute_1042, view_1125, permute_1038, view_1122, sum_351, sum_352, permute_1034, view_1119, permute_1029, view_1114, permute_1024, view_1109, permute_1013, view_1100, sum_342, sum_343, permute_1009, view_1097, permute_1005, view_1094, sum_336, sum_337, permute_1001, view_1091, permute_996, view_1086, permute_991, view_1081, permute_980, view_1072, sum_327, sum_328, permute_976, view_1069, permute_972, view_1066, sum_321, sum_322, permute_968, view_1063, permute_963, view_1058, permute_958, view_1053, permute_947, view_1044, sum_312, sum_313, permute_943, view_1041, permute_939, view_1038, sum_306, sum_307, permute_935, view_1035, permute_930, view_1030, permute_925, view_1025, permute_914, view_1016, sum_297, sum_298, permute_910, view_1013, permute_906, view_1010, sum_291, sum_292, permute_902, view_1007, permute_897, view_1002, permute_892, view_997, permute_881, view_988, sum_282, sum_283, permute_877, view_985, permute_873, view_982, sum_276, sum_277, permute_869, view_979, permute_864, view_974, permute_859, view_969, permute_848, view_960, sum_267, sum_268, permute_844, view_957, permute_840, view_954, sum_261, sum_262, permute_836, view_951, permute_831, view_946, permute_826, view_941, permute_815, view_932, sum_252, sum_253, permute_811, view_929, permute_807, view_926, sum_246, sum_247, permute_803, view_923, permute_798, view_918, permute_793, view_913, permute_782, view_904, sum_237, sum_238, permute_778, view_901, permute_774, view_898, sum_231, sum_232, permute_770, view_895, permute_765, view_890, permute_760, view_885, permute_749, view_876, sum_222, sum_223, permute_745, view_873, permute_741, view_870, sum_216, sum_217, permute_737, view_867, permute_732, view_862, permute_727, view_857, permute_716, view_848, sum_207, sum_208, permute_712, view_845, permute_708, view_842, sum_201, sum_202, permute_704, view_839, permute_699, view_834, permute_694, view_829, permute_683, view_820, sum_192, sum_193, permute_679, view_817, permute_675, view_814, sum_186, sum_187, permute_671, view_811, permute_666, view_806, permute_661, view_801, permute_650, view_792, sum_177, sum_178, permute_646, view_789, permute_642, view_786, sum_171, sum_172, permute_638, view_783, permute_633, view_778, permute_628, view_773, permute_617, view_764, sum_162, sum_163, permute_613, view_761, permute_609, view_758, sum_156, sum_157, permute_605, view_755, permute_600, view_750, permute_595, view_745, permute_584, view_736, sum_147, sum_148, permute_580, view_733, permute_576, view_730, sum_141, sum_142, permute_572, view_727, permute_567, view_722, permute_562, view_717, permute_551, view_708, sum_132, sum_133, permute_547, view_705, permute_543, view_702, sum_126, sum_127, permute_539, view_699, permute_534, view_694, permute_529, view_689, permute_518, view_680, sum_117, sum_118, permute_514, view_677, permute_510, view_674, sum_111, sum_112, permute_506, view_671, permute_501, view_666, permute_496, view_661, permute_485, view_652, sum_102, sum_103, permute_481, view_649, permute_477, view_646, sum_96, sum_97, permute_473, view_643, permute_468, view_638, permute_463, view_633, permute_452, view_624, sum_87, sum_88, permute_448, view_621, permute_444, view_618, sum_81, sum_82, permute_440, view_615, permute_435, view_610, permute_430, view_605, permute_419, view_596, sum_72, sum_73, permute_415, view_593, permute_411, view_590, sum_66, sum_67, permute_407, view_587, permute_402, view_582, permute_397, view_577, permute_386, view_568, sum_57, sum_58, permute_382, view_565, permute_378, view_562, sum_51, sum_52, permute_374, view_559, permute_369, view_554, permute_364, view_549, permute_353, view_540, sum_42, sum_43, permute_349, view_537, permute_345, view_534, sum_36, sum_37, permute_341, view_531, None, None, None, None]
    