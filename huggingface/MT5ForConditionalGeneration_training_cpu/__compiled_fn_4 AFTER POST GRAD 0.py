from __future__ import annotations



def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[512]", primals_4: "f32[512]", primals_5: "f32[512]", primals_6: "f32[512]", primals_7: "f32[512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[512]", primals_12: "f32[512]", primals_13: "f32[512]", primals_14: "f32[512]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[512]", primals_22: "f32[512]", primals_23: "f32[512]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_38: "f32[512]", primals_39: "f32[512]", primals_40: "f32[512]", primals_41: "f32[512]", primals_42: "f32[512]", primals_192: "i64[1, 128]", view: "i64[1, 128]", getitem: "f32[1, 128, 512]", getitem_1: "b8[1, 128, 512]", rsqrt: "f32[1, 128, 1]", view_1: "f32[128, 512]", add_3: "i64[128, 128]", getitem_3: "b8[1, 6, 128, 128]", view_19: "f32[128, 384]", getitem_5: "b8[1, 128, 512]", add_6: "f32[1, 128, 512]", rsqrt_1: "f32[1, 128, 1]", view_21: "f32[128, 512]", mm_4: "f32[128, 1024]", tanh: "f32[1, 128, 1024]", mm_5: "f32[128, 1024]", getitem_7: "b8[1, 128, 1024]", view_25: "f32[128, 1024]", getitem_9: "b8[1, 128, 512]", add_10: "f32[1, 128, 512]", rsqrt_2: "f32[1, 128, 1]", view_27: "f32[128, 512]", getitem_11: "b8[1, 6, 128, 128]", view_45: "f32[128, 384]", getitem_13: "b8[1, 128, 512]", add_13: "f32[1, 128, 512]", rsqrt_3: "f32[1, 128, 1]", view_47: "f32[128, 512]", mm_11: "f32[128, 1024]", tanh_1: "f32[1, 128, 1024]", mm_12: "f32[128, 1024]", getitem_15: "b8[1, 128, 1024]", view_51: "f32[128, 1024]", getitem_17: "b8[1, 128, 512]", add_17: "f32[1, 128, 512]", rsqrt_4: "f32[1, 128, 1]", view_53: "f32[128, 512]", getitem_19: "b8[1, 6, 128, 128]", view_71: "f32[128, 384]", getitem_21: "b8[1, 128, 512]", add_20: "f32[1, 128, 512]", rsqrt_5: "f32[1, 128, 1]", view_73: "f32[128, 512]", mm_18: "f32[128, 1024]", tanh_2: "f32[1, 128, 1024]", mm_19: "f32[128, 1024]", getitem_23: "b8[1, 128, 1024]", view_77: "f32[128, 1024]", getitem_25: "b8[1, 128, 512]", add_24: "f32[1, 128, 512]", rsqrt_6: "f32[1, 128, 1]", view_79: "f32[128, 512]", getitem_27: "b8[1, 6, 128, 128]", view_97: "f32[128, 384]", getitem_29: "b8[1, 128, 512]", add_27: "f32[1, 128, 512]", rsqrt_7: "f32[1, 128, 1]", view_99: "f32[128, 512]", mm_25: "f32[128, 1024]", tanh_3: "f32[1, 128, 1024]", mm_26: "f32[128, 1024]", getitem_31: "b8[1, 128, 1024]", view_103: "f32[128, 1024]", getitem_33: "b8[1, 128, 512]", add_31: "f32[1, 128, 512]", rsqrt_8: "f32[1, 128, 1]", view_105: "f32[128, 512]", getitem_35: "b8[1, 6, 128, 128]", view_123: "f32[128, 384]", getitem_37: "b8[1, 128, 512]", add_34: "f32[1, 128, 512]", rsqrt_9: "f32[1, 128, 1]", view_125: "f32[128, 512]", mm_32: "f32[128, 1024]", tanh_4: "f32[1, 128, 1024]", mm_33: "f32[128, 1024]", getitem_39: "b8[1, 128, 1024]", view_129: "f32[128, 1024]", getitem_41: "b8[1, 128, 512]", add_38: "f32[1, 128, 512]", rsqrt_10: "f32[1, 128, 1]", view_131: "f32[128, 512]", getitem_43: "b8[1, 6, 128, 128]", view_149: "f32[128, 384]", getitem_45: "b8[1, 128, 512]", add_41: "f32[1, 128, 512]", rsqrt_11: "f32[1, 128, 1]", view_151: "f32[128, 512]", mm_39: "f32[128, 1024]", tanh_5: "f32[1, 128, 1024]", mm_40: "f32[128, 1024]", getitem_47: "b8[1, 128, 1024]", view_155: "f32[128, 1024]", getitem_49: "b8[1, 128, 512]", add_45: "f32[1, 128, 512]", rsqrt_12: "f32[1, 128, 1]", view_157: "f32[128, 512]", getitem_51: "b8[1, 6, 128, 128]", view_175: "f32[128, 384]", getitem_53: "b8[1, 128, 512]", add_48: "f32[1, 128, 512]", rsqrt_13: "f32[1, 128, 1]", view_177: "f32[128, 512]", mm_46: "f32[128, 1024]", tanh_6: "f32[1, 128, 1024]", mm_47: "f32[128, 1024]", getitem_55: "b8[1, 128, 1024]", view_181: "f32[128, 1024]", getitem_57: "b8[1, 128, 512]", add_52: "f32[1, 128, 512]", rsqrt_14: "f32[1, 128, 1]", view_183: "f32[128, 512]", getitem_59: "b8[1, 6, 128, 128]", view_201: "f32[128, 384]", getitem_61: "b8[1, 128, 512]", add_55: "f32[1, 128, 512]", rsqrt_15: "f32[1, 128, 1]", view_203: "f32[128, 512]", mm_53: "f32[128, 1024]", tanh_7: "f32[1, 128, 1024]", mm_54: "f32[128, 1024]", getitem_63: "b8[1, 128, 1024]", view_207: "f32[128, 1024]", getitem_65: "b8[1, 128, 512]", add_59: "f32[1, 128, 512]", rsqrt_16: "f32[1, 128, 1]", getitem_67: "b8[1, 128, 512]", view_209: "i64[1, 128]", getitem_68: "f32[1, 128, 512]", getitem_69: "b8[1, 128, 512]", rsqrt_17: "f32[1, 128, 1]", view_210: "f32[128, 512]", add_63: "i64[128, 128]", getitem_71: "b8[1, 6, 128, 128]", view_228: "f32[128, 384]", getitem_73: "b8[1, 128, 512]", add_66: "f32[1, 128, 512]", rsqrt_18: "f32[1, 128, 1]", view_230: "f32[128, 512]", view_233: "f32[128, 512]", getitem_75: "b8[1, 6, 128, 128]", view_248: "f32[128, 384]", getitem_77: "b8[1, 128, 512]", add_70: "f32[1, 128, 512]", rsqrt_19: "f32[1, 128, 1]", view_250: "f32[128, 512]", mm_64: "f32[128, 1024]", tanh_8: "f32[1, 128, 1024]", mm_65: "f32[128, 1024]", getitem_79: "b8[1, 128, 1024]", view_254: "f32[128, 1024]", getitem_81: "b8[1, 128, 512]", add_74: "f32[1, 128, 512]", rsqrt_20: "f32[1, 128, 1]", view_256: "f32[128, 512]", getitem_83: "b8[1, 6, 128, 128]", view_274: "f32[128, 384]", getitem_85: "b8[1, 128, 512]", add_77: "f32[1, 128, 512]", rsqrt_21: "f32[1, 128, 1]", view_276: "f32[128, 512]", getitem_87: "b8[1, 6, 128, 128]", view_294: "f32[128, 384]", getitem_89: "b8[1, 128, 512]", add_80: "f32[1, 128, 512]", rsqrt_22: "f32[1, 128, 1]", view_296: "f32[128, 512]", mm_75: "f32[128, 1024]", tanh_9: "f32[1, 128, 1024]", mm_76: "f32[128, 1024]", getitem_91: "b8[1, 128, 1024]", view_300: "f32[128, 1024]", getitem_93: "b8[1, 128, 512]", add_84: "f32[1, 128, 512]", rsqrt_23: "f32[1, 128, 1]", view_302: "f32[128, 512]", getitem_95: "b8[1, 6, 128, 128]", view_320: "f32[128, 384]", getitem_97: "b8[1, 128, 512]", add_87: "f32[1, 128, 512]", rsqrt_24: "f32[1, 128, 1]", view_322: "f32[128, 512]", getitem_99: "b8[1, 6, 128, 128]", view_340: "f32[128, 384]", getitem_101: "b8[1, 128, 512]", add_90: "f32[1, 128, 512]", rsqrt_25: "f32[1, 128, 1]", view_342: "f32[128, 512]", mm_86: "f32[128, 1024]", tanh_10: "f32[1, 128, 1024]", mm_87: "f32[128, 1024]", getitem_103: "b8[1, 128, 1024]", view_346: "f32[128, 1024]", getitem_105: "b8[1, 128, 512]", add_94: "f32[1, 128, 512]", rsqrt_26: "f32[1, 128, 1]", view_348: "f32[128, 512]", getitem_107: "b8[1, 6, 128, 128]", view_366: "f32[128, 384]", getitem_109: "b8[1, 128, 512]", add_97: "f32[1, 128, 512]", rsqrt_27: "f32[1, 128, 1]", view_368: "f32[128, 512]", getitem_111: "b8[1, 6, 128, 128]", view_386: "f32[128, 384]", getitem_113: "b8[1, 128, 512]", add_100: "f32[1, 128, 512]", rsqrt_28: "f32[1, 128, 1]", view_388: "f32[128, 512]", mm_97: "f32[128, 1024]", tanh_11: "f32[1, 128, 1024]", mm_98: "f32[128, 1024]", getitem_115: "b8[1, 128, 1024]", view_392: "f32[128, 1024]", getitem_117: "b8[1, 128, 512]", add_104: "f32[1, 128, 512]", rsqrt_29: "f32[1, 128, 1]", view_394: "f32[128, 512]", getitem_119: "b8[1, 6, 128, 128]", view_412: "f32[128, 384]", getitem_121: "b8[1, 128, 512]", add_107: "f32[1, 128, 512]", rsqrt_30: "f32[1, 128, 1]", view_414: "f32[128, 512]", getitem_123: "b8[1, 6, 128, 128]", view_432: "f32[128, 384]", getitem_125: "b8[1, 128, 512]", add_110: "f32[1, 128, 512]", rsqrt_31: "f32[1, 128, 1]", view_434: "f32[128, 512]", mm_108: "f32[128, 1024]", tanh_12: "f32[1, 128, 1024]", mm_109: "f32[128, 1024]", getitem_127: "b8[1, 128, 1024]", view_438: "f32[128, 1024]", getitem_129: "b8[1, 128, 512]", add_114: "f32[1, 128, 512]", rsqrt_32: "f32[1, 128, 1]", view_440: "f32[128, 512]", getitem_131: "b8[1, 6, 128, 128]", view_458: "f32[128, 384]", getitem_133: "b8[1, 128, 512]", add_117: "f32[1, 128, 512]", rsqrt_33: "f32[1, 128, 1]", view_460: "f32[128, 512]", getitem_135: "b8[1, 6, 128, 128]", view_478: "f32[128, 384]", getitem_137: "b8[1, 128, 512]", add_120: "f32[1, 128, 512]", rsqrt_34: "f32[1, 128, 1]", view_480: "f32[128, 512]", mm_119: "f32[128, 1024]", tanh_13: "f32[1, 128, 1024]", mm_120: "f32[128, 1024]", getitem_139: "b8[1, 128, 1024]", view_484: "f32[128, 1024]", getitem_141: "b8[1, 128, 512]", add_124: "f32[1, 128, 512]", rsqrt_35: "f32[1, 128, 1]", view_486: "f32[128, 512]", getitem_143: "b8[1, 6, 128, 128]", view_504: "f32[128, 384]", getitem_145: "b8[1, 128, 512]", add_127: "f32[1, 128, 512]", rsqrt_36: "f32[1, 128, 1]", view_506: "f32[128, 512]", getitem_147: "b8[1, 6, 128, 128]", view_524: "f32[128, 384]", getitem_149: "b8[1, 128, 512]", add_130: "f32[1, 128, 512]", rsqrt_37: "f32[1, 128, 1]", view_526: "f32[128, 512]", mm_130: "f32[128, 1024]", tanh_14: "f32[1, 128, 1024]", mm_131: "f32[128, 1024]", getitem_151: "b8[1, 128, 1024]", view_530: "f32[128, 1024]", getitem_153: "b8[1, 128, 512]", add_134: "f32[1, 128, 512]", rsqrt_38: "f32[1, 128, 1]", view_532: "f32[128, 512]", getitem_155: "b8[1, 6, 128, 128]", view_550: "f32[128, 384]", getitem_157: "b8[1, 128, 512]", add_137: "f32[1, 128, 512]", rsqrt_39: "f32[1, 128, 1]", view_552: "f32[128, 512]", getitem_159: "b8[1, 6, 128, 128]", view_570: "f32[128, 384]", getitem_161: "b8[1, 128, 512]", add_140: "f32[1, 128, 512]", rsqrt_40: "f32[1, 128, 1]", view_572: "f32[128, 512]", mm_141: "f32[128, 1024]", tanh_15: "f32[1, 128, 1024]", mm_142: "f32[128, 1024]", getitem_163: "b8[1, 128, 1024]", view_576: "f32[128, 1024]", getitem_165: "b8[1, 128, 512]", add_144: "f32[1, 128, 512]", rsqrt_41: "f32[1, 128, 1]", getitem_167: "b8[1, 128, 512]", view_578: "f32[128, 512]", sub_30: "f32[128, 250112]", convert_element_type_7: "f32[]", permute_269: "f32[250112, 512]", permute_273: "f32[512, 1024]", permute_277: "f32[1024, 512]", permute_281: "f32[1024, 512]", permute_285: "f32[512, 384]", permute_288: "f32[6, 128, 128]", permute_289: "f32[6, 64, 128]", alias_87: "f32[1, 6, 128, 128]", permute_290: "f32[6, 64, 128]", permute_291: "f32[6, 128, 64]", permute_296: "f32[384, 512]", permute_301: "f32[384, 512]", permute_306: "f32[384, 512]", permute_310: "f32[512, 384]", permute_313: "f32[6, 128, 128]", permute_314: "f32[6, 64, 128]", alias_89: "f32[1, 6, 128, 128]", permute_315: "f32[6, 64, 128]", permute_316: "f32[6, 128, 64]", permute_321: "f32[384, 512]", permute_326: "f32[384, 512]", permute_331: "f32[384, 512]", permute_335: "f32[512, 1024]", permute_339: "f32[1024, 512]", permute_343: "f32[1024, 512]", permute_347: "f32[512, 384]", permute_350: "f32[6, 128, 128]", permute_351: "f32[6, 64, 128]", alias_93: "f32[1, 6, 128, 128]", permute_352: "f32[6, 64, 128]", permute_353: "f32[6, 128, 64]", permute_358: "f32[384, 512]", permute_363: "f32[384, 512]", permute_368: "f32[384, 512]", permute_372: "f32[512, 384]", permute_375: "f32[6, 128, 128]", permute_376: "f32[6, 64, 128]", alias_95: "f32[1, 6, 128, 128]", permute_377: "f32[6, 64, 128]", permute_378: "f32[6, 128, 64]", permute_383: "f32[384, 512]", permute_388: "f32[384, 512]", permute_393: "f32[384, 512]", permute_397: "f32[512, 1024]", permute_401: "f32[1024, 512]", permute_405: "f32[1024, 512]", permute_409: "f32[512, 384]", permute_412: "f32[6, 128, 128]", permute_413: "f32[6, 64, 128]", alias_99: "f32[1, 6, 128, 128]", permute_414: "f32[6, 64, 128]", permute_415: "f32[6, 128, 64]", permute_420: "f32[384, 512]", permute_425: "f32[384, 512]", permute_430: "f32[384, 512]", permute_434: "f32[512, 384]", permute_437: "f32[6, 128, 128]", permute_438: "f32[6, 64, 128]", alias_101: "f32[1, 6, 128, 128]", permute_439: "f32[6, 64, 128]", permute_440: "f32[6, 128, 64]", permute_445: "f32[384, 512]", permute_450: "f32[384, 512]", permute_455: "f32[384, 512]", permute_459: "f32[512, 1024]", permute_463: "f32[1024, 512]", permute_467: "f32[1024, 512]", permute_471: "f32[512, 384]", permute_474: "f32[6, 128, 128]", permute_475: "f32[6, 64, 128]", alias_105: "f32[1, 6, 128, 128]", permute_476: "f32[6, 64, 128]", permute_477: "f32[6, 128, 64]", permute_482: "f32[384, 512]", permute_487: "f32[384, 512]", permute_492: "f32[384, 512]", permute_496: "f32[512, 384]", permute_499: "f32[6, 128, 128]", permute_500: "f32[6, 64, 128]", alias_107: "f32[1, 6, 128, 128]", permute_501: "f32[6, 64, 128]", permute_502: "f32[6, 128, 64]", permute_507: "f32[384, 512]", permute_512: "f32[384, 512]", permute_517: "f32[384, 512]", permute_521: "f32[512, 1024]", permute_525: "f32[1024, 512]", permute_529: "f32[1024, 512]", permute_533: "f32[512, 384]", permute_536: "f32[6, 128, 128]", permute_537: "f32[6, 64, 128]", alias_111: "f32[1, 6, 128, 128]", permute_538: "f32[6, 64, 128]", permute_539: "f32[6, 128, 64]", permute_544: "f32[384, 512]", permute_549: "f32[384, 512]", permute_554: "f32[384, 512]", permute_558: "f32[512, 384]", permute_561: "f32[6, 128, 128]", permute_562: "f32[6, 64, 128]", alias_113: "f32[1, 6, 128, 128]", permute_563: "f32[6, 64, 128]", permute_564: "f32[6, 128, 64]", permute_569: "f32[384, 512]", permute_574: "f32[384, 512]", permute_579: "f32[384, 512]", permute_583: "f32[512, 1024]", permute_587: "f32[1024, 512]", permute_591: "f32[1024, 512]", permute_595: "f32[512, 384]", permute_598: "f32[6, 128, 128]", permute_599: "f32[6, 64, 128]", alias_117: "f32[1, 6, 128, 128]", permute_600: "f32[6, 64, 128]", permute_601: "f32[6, 128, 64]", permute_606: "f32[384, 512]", permute_611: "f32[384, 512]", permute_616: "f32[384, 512]", permute_620: "f32[512, 384]", permute_623: "f32[6, 128, 128]", permute_624: "f32[6, 64, 128]", alias_119: "f32[1, 6, 128, 128]", permute_625: "f32[6, 64, 128]", permute_626: "f32[6, 128, 64]", permute_631: "f32[384, 512]", permute_636: "f32[384, 512]", permute_641: "f32[384, 512]", permute_645: "f32[512, 1024]", permute_649: "f32[1024, 512]", permute_653: "f32[1024, 512]", permute_657: "f32[512, 384]", permute_660: "f32[6, 128, 128]", permute_661: "f32[6, 64, 128]", alias_123: "f32[1, 6, 128, 128]", permute_662: "f32[6, 64, 128]", permute_663: "f32[6, 128, 64]", permute_668: "f32[384, 512]", permute_673: "f32[384, 512]", permute_678: "f32[384, 512]", permute_682: "f32[512, 384]", permute_685: "f32[6, 128, 128]", permute_686: "f32[6, 64, 128]", alias_125: "f32[1, 6, 128, 128]", permute_687: "f32[6, 64, 128]", permute_688: "f32[6, 128, 64]", permute_693: "f32[384, 512]", permute_698: "f32[384, 512]", permute_703: "f32[384, 512]", permute_707: "f32[512, 1024]", permute_711: "f32[1024, 512]", permute_715: "f32[1024, 512]", permute_719: "f32[512, 384]", permute_722: "f32[6, 128, 128]", permute_723: "f32[6, 64, 128]", alias_129: "f32[1, 6, 128, 128]", permute_724: "f32[6, 64, 128]", permute_725: "f32[6, 128, 64]", permute_730: "f32[384, 512]", permute_735: "f32[384, 512]", permute_740: "f32[384, 512]", permute_744: "f32[512, 384]", permute_747: "f32[6, 128, 128]", permute_748: "f32[6, 64, 128]", alias_131: "f32[1, 6, 128, 128]", permute_750: "f32[6, 64, 128]", permute_751: "f32[6, 128, 64]", permute_756: "f32[384, 512]", permute_761: "f32[384, 512]", permute_766: "f32[384, 512]", permute_770: "f32[512, 1024]", permute_774: "f32[1024, 512]", permute_778: "f32[1024, 512]", permute_782: "f32[512, 384]", permute_785: "f32[6, 128, 128]", permute_786: "f32[6, 64, 128]", alias_136: "f32[1, 6, 128, 128]", permute_787: "f32[6, 64, 128]", permute_788: "f32[6, 128, 64]", permute_793: "f32[384, 512]", permute_798: "f32[384, 512]", permute_803: "f32[384, 512]", permute_807: "f32[512, 1024]", permute_811: "f32[1024, 512]", permute_815: "f32[1024, 512]", permute_819: "f32[512, 384]", permute_822: "f32[6, 128, 128]", permute_823: "f32[6, 64, 128]", alias_140: "f32[1, 6, 128, 128]", permute_824: "f32[6, 64, 128]", permute_825: "f32[6, 128, 64]", permute_830: "f32[384, 512]", permute_835: "f32[384, 512]", permute_840: "f32[384, 512]", permute_844: "f32[512, 1024]", permute_848: "f32[1024, 512]", permute_852: "f32[1024, 512]", permute_856: "f32[512, 384]", permute_859: "f32[6, 128, 128]", permute_860: "f32[6, 64, 128]", alias_144: "f32[1, 6, 128, 128]", permute_861: "f32[6, 64, 128]", permute_862: "f32[6, 128, 64]", permute_867: "f32[384, 512]", permute_872: "f32[384, 512]", permute_877: "f32[384, 512]", permute_881: "f32[512, 1024]", permute_885: "f32[1024, 512]", permute_889: "f32[1024, 512]", permute_893: "f32[512, 384]", permute_896: "f32[6, 128, 128]", permute_897: "f32[6, 64, 128]", alias_148: "f32[1, 6, 128, 128]", permute_898: "f32[6, 64, 128]", permute_899: "f32[6, 128, 64]", permute_904: "f32[384, 512]", permute_909: "f32[384, 512]", permute_914: "f32[384, 512]", permute_918: "f32[512, 1024]", permute_922: "f32[1024, 512]", permute_926: "f32[1024, 512]", permute_930: "f32[512, 384]", permute_933: "f32[6, 128, 128]", permute_934: "f32[6, 64, 128]", alias_152: "f32[1, 6, 128, 128]", permute_935: "f32[6, 64, 128]", permute_936: "f32[6, 128, 64]", permute_941: "f32[384, 512]", permute_946: "f32[384, 512]", permute_951: "f32[384, 512]", permute_955: "f32[512, 1024]", permute_959: "f32[1024, 512]", permute_963: "f32[1024, 512]", permute_967: "f32[512, 384]", permute_970: "f32[6, 128, 128]", permute_971: "f32[6, 64, 128]", alias_156: "f32[1, 6, 128, 128]", permute_972: "f32[6, 64, 128]", permute_973: "f32[6, 128, 64]", permute_978: "f32[384, 512]", permute_983: "f32[384, 512]", permute_988: "f32[384, 512]", permute_992: "f32[512, 1024]", permute_996: "f32[1024, 512]", permute_1000: "f32[1024, 512]", permute_1004: "f32[512, 384]", permute_1007: "f32[6, 128, 128]", permute_1008: "f32[6, 64, 128]", alias_160: "f32[1, 6, 128, 128]", permute_1009: "f32[6, 64, 128]", permute_1010: "f32[6, 128, 64]", permute_1015: "f32[384, 512]", permute_1020: "f32[384, 512]", permute_1025: "f32[384, 512]", permute_1029: "f32[512, 1024]", permute_1033: "f32[1024, 512]", permute_1037: "f32[1024, 512]", permute_1041: "f32[512, 384]", permute_1044: "f32[6, 128, 128]", permute_1045: "f32[6, 64, 128]", alias_164: "f32[1, 6, 128, 128]", permute_1047: "f32[6, 64, 128]", permute_1048: "f32[6, 128, 64]", permute_1053: "f32[384, 512]", permute_1058: "f32[384, 512]", permute_1063: "f32[384, 512]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 250112]", tangents_3: "f32[1, 6, 128, 64]", tangents_4: "f32[1, 6, 128, 64]", tangents_5: "f32[1, 6, 128, 64]", tangents_6: "f32[1, 6, 128, 64]", tangents_7: "f32[1, 6, 128, 64]", tangents_8: "f32[1, 6, 128, 64]", tangents_9: "f32[1, 6, 128, 64]", tangents_10: "f32[1, 6, 128, 64]", tangents_11: "f32[1, 6, 128, 64]", tangents_12: "f32[1, 6, 128, 64]", tangents_13: "f32[1, 6, 128, 64]", tangents_14: "f32[1, 6, 128, 64]", tangents_15: "f32[1, 6, 128, 64]", tangents_16: "f32[1, 6, 128, 64]", tangents_17: "f32[1, 6, 128, 64]", tangents_18: "f32[1, 6, 128, 64]", tangents_19: "f32[1, 6, 128, 64]", tangents_20: "f32[1, 6, 128, 64]", tangents_21: "f32[1, 6, 128, 64]", tangents_22: "f32[1, 6, 128, 64]", tangents_23: "f32[1, 6, 128, 64]", tangents_24: "f32[1, 6, 128, 64]", tangents_25: "f32[1, 6, 128, 64]", tangents_26: "f32[1, 6, 128, 64]", tangents_27: "f32[1, 6, 128, 64]", tangents_28: "f32[1, 6, 128, 64]", tangents_29: "f32[1, 6, 128, 64]", tangents_30: "f32[1, 6, 128, 64]", tangents_31: "f32[1, 6, 128, 64]", tangents_32: "f32[1, 6, 128, 64]", tangents_33: "f32[1, 6, 128, 64]", tangents_34: "f32[1, 6, 128, 64]", tangents_35: "f32[1, 128, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(getitem, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    add_9: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, add_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 1024]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_14: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_48: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 1024]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_16, add_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_50: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_21: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_23: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, add_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 1024]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_30: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_32: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_100: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 1024]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_100, 0.5)
    add_30: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_34, add_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_39: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_126: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_43, add_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_128: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 1024]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_48: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_38, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_152: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 1024]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_55: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, add_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_154: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_45, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_59: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 1024]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    add_51: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_64: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_180: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 1024]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_66: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_68: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 1024]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_70: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_70, add_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 1024]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_75: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_80: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(getitem_68, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_83: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_85: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_251: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_87: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_87, add_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_253: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 1024]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_92: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_94: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_77, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_96: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_80, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_297: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 1024]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
    add_83: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, add_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_299: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_103: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_87, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_107: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_343: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    add_93: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, add_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_345: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 1024]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_114: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_116: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_97, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_118: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_100, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_389: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 1024]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
    add_103: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, add_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_391: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_125: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_104, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_127: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_107, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_435: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 1024]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_435, 0.5)
    add_113: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_131, add_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_437: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 1024]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_136: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_114, rsqrt_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_138: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, rsqrt_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_140: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_120, rsqrt_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_481: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 1024]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_142: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    add_123: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_142, add_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_483: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_120, [1, 128, 1024]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_147: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_149: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_127, rsqrt_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_151: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_527: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_130, [1, 128, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    add_133: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_153, add_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_529: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 1024]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_158: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_134, rsqrt_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_160: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, rsqrt_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_162: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_140, rsqrt_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_573: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [1, 128, 1024]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_573, 0.5)
    add_143: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
    mul_167: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, add_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_575: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_142, [1, 128, 1024]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_144, rsqrt_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_581: "i64[128]" = torch.ops.aten.reshape.default(primals_192, [-1]);  primals_192 = None
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_7);  tangents_1 = convert_element_type_7 = None
    unsqueeze_18: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_581, 1);  view_581 = None
    ne_3: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_18, -100)
    where_4: "i64[128, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_18, full_default_6);  unsqueeze_18 = full_default_6 = None
    full_default_9: "f32[128, 250112]" = torch.ops.aten.full.default([128, 250112], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[128, 250112]" = torch.ops.aten.scatter.value(full_default_9, 1, where_4, -1.0);  full_default_9 = where_4 = None
    where_5: "f32[128, 1]" = torch.ops.aten.where.self(ne_3, div_29, full_default_7);  ne_3 = div_29 = None
    mul_171: "f32[128, 250112]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    exp_25: "f32[128, 250112]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_28: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [1], True)
    mul_172: "f32[128, 250112]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_31: "f32[128, 250112]" = torch.ops.aten.sub.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    view_582: "f32[1, 128, 250112]" = torch.ops.aten.reshape.default(sub_31, [1, 128, 250112]);  sub_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    add_146: "f32[1, 128, 250112]" = torch.ops.aten.add.Tensor(tangents_2, view_582);  tangents_2 = view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1799, code: lm_logits = self.lm_head(sequence_output)
    view_583: "f32[128, 250112]" = torch.ops.aten.reshape.default(add_146, [128, 250112]);  add_146 = None
    permute_267: "f32[250112, 128]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_145: "f32[250112, 512]" = torch.ops.aten.mm.default(permute_267, view_578);  permute_267 = view_578 = None
    permute_268: "f32[512, 250112]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    mm_146: "f32[128, 512]" = torch.ops.aten.mm.default(view_583, permute_269);  view_583 = permute_269 = None
    view_584: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_146, [1, 128, 512]);  mm_146 = None
    permute_270: "f32[250112, 512]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
    mul_173: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_174: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_584, mul_173);  view_584 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_175: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_174, primals_42);  primals_42 = None
    mul_176: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_174, mul_169);  mul_174 = mul_169 = None
    sum_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1], True);  mul_176 = None
    view_585: "f32[512]" = torch.ops.aten.reshape.default(sum_29, [512]);  sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_177: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_175, add_144)
    mul_178: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_175, rsqrt_41);  mul_175 = None
    sum_30: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
    pow_59: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_41, 3);  rsqrt_41 = None
    mul_179: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
    mul_180: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_179, pow_59);  mul_179 = pow_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_180, [1, 128, 512]);  mul_180 = None
    div_30: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
    pow_60: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_144, 1.0);  add_144 = None
    mul_181: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
    mul_182: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_181);  div_30 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_147: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_9: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
    mul_183: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_184: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_147, mul_183);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_586: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_184, [128, 512]);  mul_184 = None
    permute_271: "f32[512, 128]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_147: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_271, view_576);  permute_271 = view_576 = None
    permute_272: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    mm_148: "f32[128, 1024]" = torch.ops.aten.mm.default(view_586, permute_273);  view_586 = permute_273 = None
    view_587: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_148, [1, 128, 1024]);  mm_148 = None
    permute_274: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_163, torch.float32);  getitem_163 = None
    mul_185: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_186: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_587, mul_185);  view_587 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_187: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_186, mul_167);  mul_167 = None
    mul_188: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_186, view_575);  mul_186 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_588: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_187, [128, 1024]);  mul_187 = None
    permute_275: "f32[1024, 128]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_149: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_275, view_572);  permute_275 = None
    permute_276: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    mm_150: "f32[128, 512]" = torch.ops.aten.mm.default(view_588, permute_277);  view_588 = permute_277 = None
    view_589: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_150, [1, 128, 512]);  mm_150 = None
    permute_278: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_189: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_188, mul_164);  mul_164 = None
    mul_190: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_188, add_143);  mul_188 = add_143 = None
    mul_191: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
    sub_32: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_191);  mul_191 = None
    mul_192: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_189, sub_32);  mul_189 = sub_32 = None
    mul_193: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_192, 0.7978845608028654);  mul_192 = None
    mul_194: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_193, 0.044715)
    pow_61: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_573, 2.0);  view_573 = None
    mul_195: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_61, 3.0);  pow_61 = None
    mul_196: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_193, mul_196);  mul_193 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_197: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_190, 0.5);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_149: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_148, mul_197);  add_148 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_590: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_149, [128, 1024]);  add_149 = None
    permute_279: "f32[1024, 128]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_151: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_279, view_572);  permute_279 = view_572 = None
    permute_280: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    mm_152: "f32[128, 512]" = torch.ops.aten.mm.default(view_590, permute_281);  view_590 = permute_281 = None
    view_591: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_152, [1, 128, 512]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_150: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_589, view_591);  view_589 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_282: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_198: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_150, primals_41);  primals_41 = None
    mul_199: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_150, mul_162);  add_150 = mul_162 = None
    sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1], True);  mul_199 = None
    view_592: "f32[512]" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_200: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_198, add_140)
    mul_201: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_198, rsqrt_40);  mul_198 = None
    sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_151: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_147, mul_201);  add_147 = mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_62: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_40, 3);  rsqrt_40 = None
    mul_202: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
    mul_203: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_202, pow_62);  mul_202 = pow_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_97: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_203, [1, 128, 512]);  mul_203 = None
    div_31: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
    pow_63: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_140, 1.0);  add_140 = None
    mul_204: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
    mul_205: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_204);  div_31 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_152: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_151, mul_205);  add_151 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_11: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
    mul_206: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_207: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_152, mul_206);  mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_593: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_207, [128, 512]);  mul_207 = None
    permute_283: "f32[512, 128]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_153: "f32[512, 384]" = torch.ops.aten.mm.default(permute_283, view_570);  permute_283 = view_570 = None
    permute_284: "f32[384, 512]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    mm_154: "f32[128, 384]" = torch.ops.aten.mm.default(view_593, permute_285);  view_593 = permute_285 = None
    view_594: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_154, [1, 128, 384]);  mm_154 = None
    permute_286: "f32[512, 384]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_595: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_594, [1, 128, 6, 64]);  view_594 = None
    permute_287: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_596: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_287, [6, 128, 64]);  permute_287 = None
    bmm_48: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_288, view_596);  permute_288 = None
    bmm_49: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_596, permute_289);  view_596 = permute_289 = None
    view_597: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_48, [1, 6, 128, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_153: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_34, view_597);  tangents_34 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_598: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_49, [1, 6, 128, 128]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_12: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None
    mul_208: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_209: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_598, mul_208);  view_598 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_210: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, alias_87);  mul_209 = None
    sum_33: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [-1], True)
    mul_211: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_87, sum_33);  alias_87 = sum_33 = None
    sub_33: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_33, 0);  sub_33 = None
    full_8: "f32[98304]" = torch.ops.aten.full.default([98304], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(full_8, [6, 128, 128], [16384, 128, 1], 0)
    as_strided_scatter: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_1, [6, 128, 128], [16384, 128, 1], 0);  squeeze_1 = None
    as_strided_3: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_3, [6, 128, 128], [16384, 128, 1])
    as_strided_5: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_3, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_1: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_3, as_strided_5, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_3 = as_strided_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_50: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_290, as_strided_scatter_1);  permute_290 = None
    bmm_51: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_1, permute_291);  as_strided_scatter_1 = permute_291 = None
    view_599: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_50, [1, 6, 64, 128]);  bmm_50 = None
    view_600: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_51, [1, 6, 128, 64]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_292: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_599, [0, 1, 3, 2]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_154: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_33, permute_292);  tangents_33 = permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_293: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_153, [0, 2, 1, 3]);  add_153 = None
    clone_30: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_601: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 384]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_602: "f32[128, 384]" = torch.ops.aten.reshape.default(view_601, [128, 384]);  view_601 = None
    permute_294: "f32[384, 128]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_155: "f32[384, 512]" = torch.ops.aten.mm.default(permute_294, view_233);  permute_294 = None
    permute_295: "f32[512, 384]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    mm_156: "f32[128, 512]" = torch.ops.aten.mm.default(view_602, permute_296);  view_602 = permute_296 = None
    view_603: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_156, [1, 128, 512]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_155: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(tangents_35, view_603);  tangents_35 = view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_297: "f32[384, 512]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_298: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_154, [0, 2, 1, 3]);  add_154 = None
    clone_31: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
    view_604: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_31, [1, 128, 384]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_605: "f32[128, 384]" = torch.ops.aten.reshape.default(view_604, [128, 384]);  view_604 = None
    permute_299: "f32[384, 128]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_157: "f32[384, 512]" = torch.ops.aten.mm.default(permute_299, view_233);  permute_299 = None
    permute_300: "f32[512, 384]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    mm_158: "f32[128, 512]" = torch.ops.aten.mm.default(view_605, permute_301);  view_605 = permute_301 = None
    view_606: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_158, [1, 128, 512]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_156: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_155, view_606);  add_155 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_302: "f32[384, 512]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_303: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
    clone_32: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    view_607: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_32, [1, 128, 384]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_608: "f32[128, 384]" = torch.ops.aten.reshape.default(view_607, [128, 384]);  view_607 = None
    permute_304: "f32[384, 128]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_159: "f32[384, 512]" = torch.ops.aten.mm.default(permute_304, view_552);  permute_304 = view_552 = None
    permute_305: "f32[512, 384]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    mm_160: "f32[128, 512]" = torch.ops.aten.mm.default(view_608, permute_306);  view_608 = permute_306 = None
    view_609: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_160, [1, 128, 512]);  mm_160 = None
    permute_307: "f32[384, 512]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_212: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_609, primals_40);  primals_40 = None
    mul_213: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_609, mul_160);  view_609 = mul_160 = None
    sum_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1], True);  mul_213 = None
    view_610: "f32[512]" = torch.ops.aten.reshape.default(sum_34, [512]);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_214: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_212, add_137)
    mul_215: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_212, rsqrt_39);  mul_212 = None
    sum_35: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_157: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_152, mul_215);  add_152 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_64: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_39, 3);  rsqrt_39 = None
    mul_216: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_35, -0.5);  sum_35 = None
    mul_217: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_216, pow_64);  mul_216 = pow_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_98: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_217, [1, 128, 512]);  mul_217 = None
    div_32: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
    pow_65: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_137, 1.0);  add_137 = None
    mul_218: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
    mul_219: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_218);  div_32 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_158: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_157, mul_219);  add_157 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_13: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_220: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_221: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_158, mul_220);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_611: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_221, [128, 512]);  mul_221 = None
    permute_308: "f32[512, 128]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_161: "f32[512, 384]" = torch.ops.aten.mm.default(permute_308, view_550);  permute_308 = view_550 = None
    permute_309: "f32[384, 512]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    mm_162: "f32[128, 384]" = torch.ops.aten.mm.default(view_611, permute_310);  view_611 = permute_310 = None
    view_612: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_162, [1, 128, 384]);  mm_162 = None
    permute_311: "f32[512, 384]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_613: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_612, [1, 128, 6, 64]);  view_612 = None
    permute_312: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_614: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_312, [6, 128, 64]);  permute_312 = None
    bmm_52: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_313, view_614);  permute_313 = None
    bmm_53: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_614, permute_314);  view_614 = permute_314 = None
    view_615: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_52, [1, 6, 128, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_159: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_32, view_615);  tangents_32 = view_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_616: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_53, [1, 6, 128, 128]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_14: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None
    mul_222: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_223: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_616, mul_222);  view_616 = mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_224: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_223, alias_89);  mul_223 = None
    sum_36: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [-1], True)
    mul_225: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_36);  alias_89 = sum_36 = None
    sub_34: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_2: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_34, 0);  sub_34 = None
    as_strided_scatter_2: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_2, [6, 128, 128], [16384, 128, 1], 0);  squeeze_2 = None
    as_strided_10: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_2, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_2 = None
    new_empty_strided_1: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_10, [6, 128, 128], [16384, 128, 1])
    as_strided_12: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_10, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_3: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_10, as_strided_12, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_54: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_315, as_strided_scatter_3);  permute_315 = None
    bmm_55: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_3, permute_316);  as_strided_scatter_3 = permute_316 = None
    view_617: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_54, [1, 6, 64, 128]);  bmm_54 = None
    view_618: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_55, [1, 6, 128, 64]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_317: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_617, [0, 1, 3, 2]);  view_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_160: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_31, permute_317);  tangents_31 = permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_318: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_159, [0, 2, 1, 3]);  add_159 = None
    clone_36: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    view_619: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_36, [1, 128, 384]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_620: "f32[128, 384]" = torch.ops.aten.reshape.default(view_619, [128, 384]);  view_619 = None
    permute_319: "f32[384, 128]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_163: "f32[384, 512]" = torch.ops.aten.mm.default(permute_319, view_532);  permute_319 = None
    permute_320: "f32[512, 384]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    mm_164: "f32[128, 512]" = torch.ops.aten.mm.default(view_620, permute_321);  view_620 = permute_321 = None
    view_621: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_164, [1, 128, 512]);  mm_164 = None
    permute_322: "f32[384, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_323: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_160, [0, 2, 1, 3]);  add_160 = None
    clone_37: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
    view_622: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_37, [1, 128, 384]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_623: "f32[128, 384]" = torch.ops.aten.reshape.default(view_622, [128, 384]);  view_622 = None
    permute_324: "f32[384, 128]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_165: "f32[384, 512]" = torch.ops.aten.mm.default(permute_324, view_532);  permute_324 = None
    permute_325: "f32[512, 384]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    mm_166: "f32[128, 512]" = torch.ops.aten.mm.default(view_623, permute_326);  view_623 = permute_326 = None
    view_624: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_166, [1, 128, 512]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_161: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_621, view_624);  view_621 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_327: "f32[384, 512]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_328: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_618, [0, 2, 1, 3]);  view_618 = None
    clone_38: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_625: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 384]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_626: "f32[128, 384]" = torch.ops.aten.reshape.default(view_625, [128, 384]);  view_625 = None
    permute_329: "f32[384, 128]" = torch.ops.aten.permute.default(view_626, [1, 0])
    mm_167: "f32[384, 512]" = torch.ops.aten.mm.default(permute_329, view_532);  permute_329 = view_532 = None
    permute_330: "f32[512, 384]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    mm_168: "f32[128, 512]" = torch.ops.aten.mm.default(view_626, permute_331);  view_626 = permute_331 = None
    view_627: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_168, [1, 128, 512]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_162: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_161, view_627);  add_161 = view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_332: "f32[384, 512]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_226: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_162, primals_39);  primals_39 = None
    mul_227: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_162, mul_158);  add_162 = mul_158 = None
    sum_37: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1], True);  mul_227 = None
    view_628: "f32[512]" = torch.ops.aten.reshape.default(sum_37, [512]);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_228: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_226, add_134)
    mul_229: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_226, rsqrt_38);  mul_226 = None
    sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_163: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_158, mul_229);  add_158 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_66: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_38, 3);  rsqrt_38 = None
    mul_230: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
    mul_231: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_230, pow_66);  mul_230 = pow_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_99: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_231, [1, 128, 512]);  mul_231 = None
    div_33: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
    pow_67: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_134, 1.0);  add_134 = None
    mul_232: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
    mul_233: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_232);  div_33 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_164: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_163, mul_233);  add_163 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_15: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
    mul_234: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_235: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_164, mul_234);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_629: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_235, [128, 512]);  mul_235 = None
    permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(view_629, [1, 0])
    mm_169: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_333, view_530);  permute_333 = view_530 = None
    permute_334: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    mm_170: "f32[128, 1024]" = torch.ops.aten.mm.default(view_629, permute_335);  view_629 = permute_335 = None
    view_630: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_170, [1, 128, 1024]);  mm_170 = None
    permute_336: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_236: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_237: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_630, mul_236);  view_630 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_238: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_237, mul_156);  mul_156 = None
    mul_239: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_237, view_529);  mul_237 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_631: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_238, [128, 1024]);  mul_238 = None
    permute_337: "f32[1024, 128]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_171: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_337, view_526);  permute_337 = None
    permute_338: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    mm_172: "f32[128, 512]" = torch.ops.aten.mm.default(view_631, permute_339);  view_631 = permute_339 = None
    view_632: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_172, [1, 128, 512]);  mm_172 = None
    permute_340: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_240: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_239, mul_153);  mul_153 = None
    mul_241: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_239, add_133);  mul_239 = add_133 = None
    mul_242: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
    sub_35: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_242);  mul_242 = None
    mul_243: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_240, sub_35);  mul_240 = sub_35 = None
    mul_244: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_243, 0.7978845608028654);  mul_243 = None
    mul_245: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_244, 0.044715)
    pow_68: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 2.0);  view_527 = None
    mul_246: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_68, 3.0);  pow_68 = None
    mul_247: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_165: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_244, mul_247);  mul_244 = mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_248: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_241, 0.5);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_166: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_165, mul_248);  add_165 = mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_633: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_166, [128, 1024]);  add_166 = None
    permute_341: "f32[1024, 128]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_173: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_341, view_526);  permute_341 = view_526 = None
    permute_342: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    mm_174: "f32[128, 512]" = torch.ops.aten.mm.default(view_633, permute_343);  view_633 = permute_343 = None
    view_634: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_174, [1, 128, 512]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_167: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_632, view_634);  view_632 = view_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_344: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_249: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_167, primals_38);  primals_38 = None
    mul_250: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_167, mul_151);  add_167 = mul_151 = None
    sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1], True);  mul_250 = None
    view_635: "f32[512]" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_251: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_249, add_130)
    mul_252: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_249, rsqrt_37);  mul_249 = None
    sum_40: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_168: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_164, mul_252);  add_164 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_69: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_37, 3);  rsqrt_37 = None
    mul_253: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
    mul_254: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_253, pow_69);  mul_253 = pow_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_100: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_254, [1, 128, 512]);  mul_254 = None
    div_34: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
    pow_70: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_130, 1.0);  add_130 = None
    mul_255: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
    mul_256: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_255);  div_34 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_169: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_168, mul_256);  add_168 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_17: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
    mul_257: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_258: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_169, mul_257);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_636: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_258, [128, 512]);  mul_258 = None
    permute_345: "f32[512, 128]" = torch.ops.aten.permute.default(view_636, [1, 0])
    mm_175: "f32[512, 384]" = torch.ops.aten.mm.default(permute_345, view_524);  permute_345 = view_524 = None
    permute_346: "f32[384, 512]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    mm_176: "f32[128, 384]" = torch.ops.aten.mm.default(view_636, permute_347);  view_636 = permute_347 = None
    view_637: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_176, [1, 128, 384]);  mm_176 = None
    permute_348: "f32[512, 384]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_638: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_637, [1, 128, 6, 64]);  view_637 = None
    permute_349: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_639: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_349, [6, 128, 64]);  permute_349 = None
    bmm_56: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_350, view_639);  permute_350 = None
    bmm_57: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_639, permute_351);  view_639 = permute_351 = None
    view_640: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_56, [1, 6, 128, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_170: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_30, view_640);  tangents_30 = view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_641: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [1, 6, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_18: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_259: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_260: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_641, mul_259);  view_641 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_261: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_260, alias_93);  mul_260 = None
    sum_41: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [-1], True)
    mul_262: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_93, sum_41);  alias_93 = sum_41 = None
    sub_36: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_3: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_36, 0);  sub_36 = None
    as_strided_scatter_4: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_3, [6, 128, 128], [16384, 128, 1], 0);  squeeze_3 = None
    as_strided_17: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_4 = None
    new_empty_strided_2: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_17, [6, 128, 128], [16384, 128, 1])
    as_strided_19: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_17, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_5: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_17, as_strided_19, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_17 = as_strided_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_58: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_352, as_strided_scatter_5);  permute_352 = None
    bmm_59: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_5, permute_353);  as_strided_scatter_5 = permute_353 = None
    view_642: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_58, [1, 6, 64, 128]);  bmm_58 = None
    view_643: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_59, [1, 6, 128, 64]);  bmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_354: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_642, [0, 1, 3, 2]);  view_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_171: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_29, permute_354);  tangents_29 = permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_355: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
    clone_44: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_644: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_44, [1, 128, 384]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_645: "f32[128, 384]" = torch.ops.aten.reshape.default(view_644, [128, 384]);  view_644 = None
    permute_356: "f32[384, 128]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_177: "f32[384, 512]" = torch.ops.aten.mm.default(permute_356, view_233);  permute_356 = None
    permute_357: "f32[512, 384]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    mm_178: "f32[128, 512]" = torch.ops.aten.mm.default(view_645, permute_358);  view_645 = permute_358 = None
    view_646: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_178, [1, 128, 512]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_172: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_156, view_646);  add_156 = view_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_359: "f32[384, 512]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_360: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_171, [0, 2, 1, 3]);  add_171 = None
    clone_45: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_647: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_45, [1, 128, 384]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_648: "f32[128, 384]" = torch.ops.aten.reshape.default(view_647, [128, 384]);  view_647 = None
    permute_361: "f32[384, 128]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_179: "f32[384, 512]" = torch.ops.aten.mm.default(permute_361, view_233);  permute_361 = None
    permute_362: "f32[512, 384]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    mm_180: "f32[128, 512]" = torch.ops.aten.mm.default(view_648, permute_363);  view_648 = permute_363 = None
    view_649: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_180, [1, 128, 512]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_173: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_172, view_649);  add_172 = view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_364: "f32[384, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_365: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_643, [0, 2, 1, 3]);  view_643 = None
    clone_46: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
    view_650: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 384]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_651: "f32[128, 384]" = torch.ops.aten.reshape.default(view_650, [128, 384]);  view_650 = None
    permute_366: "f32[384, 128]" = torch.ops.aten.permute.default(view_651, [1, 0])
    mm_181: "f32[384, 512]" = torch.ops.aten.mm.default(permute_366, view_506);  permute_366 = view_506 = None
    permute_367: "f32[512, 384]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    mm_182: "f32[128, 512]" = torch.ops.aten.mm.default(view_651, permute_368);  view_651 = permute_368 = None
    view_652: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_182, [1, 128, 512]);  mm_182 = None
    permute_369: "f32[384, 512]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_263: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_652, primals_37);  primals_37 = None
    mul_264: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_652, mul_149);  view_652 = mul_149 = None
    sum_42: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1], True);  mul_264 = None
    view_653: "f32[512]" = torch.ops.aten.reshape.default(sum_42, [512]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_265: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_263, add_127)
    mul_266: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_263, rsqrt_36);  mul_263 = None
    sum_43: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_174: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_169, mul_266);  add_169 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_71: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_36, 3);  rsqrt_36 = None
    mul_267: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_43, -0.5);  sum_43 = None
    mul_268: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_267, pow_71);  mul_267 = pow_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_101: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_268, [1, 128, 512]);  mul_268 = None
    div_35: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
    pow_72: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_127, 1.0);  add_127 = None
    mul_269: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
    mul_270: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_269);  div_35 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_175: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_174, mul_270);  add_174 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_19: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
    mul_271: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_272: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_175, mul_271);  mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_654: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_272, [128, 512]);  mul_272 = None
    permute_370: "f32[512, 128]" = torch.ops.aten.permute.default(view_654, [1, 0])
    mm_183: "f32[512, 384]" = torch.ops.aten.mm.default(permute_370, view_504);  permute_370 = view_504 = None
    permute_371: "f32[384, 512]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    mm_184: "f32[128, 384]" = torch.ops.aten.mm.default(view_654, permute_372);  view_654 = permute_372 = None
    view_655: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_184, [1, 128, 384]);  mm_184 = None
    permute_373: "f32[512, 384]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_656: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_655, [1, 128, 6, 64]);  view_655 = None
    permute_374: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_657: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_374, [6, 128, 64]);  permute_374 = None
    bmm_60: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_375, view_657);  permute_375 = None
    bmm_61: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_657, permute_376);  view_657 = permute_376 = None
    view_658: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_60, [1, 6, 128, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_176: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_28, view_658);  tangents_28 = view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_659: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [1, 6, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_20: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
    mul_273: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_274: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_659, mul_273);  view_659 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_275: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_274, alias_95);  mul_274 = None
    sum_44: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_44);  alias_95 = sum_44 = None
    sub_37: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_4: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_37, 0);  sub_37 = None
    as_strided_scatter_6: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_4, [6, 128, 128], [16384, 128, 1], 0);  squeeze_4 = None
    as_strided_24: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_6, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_6 = None
    new_empty_strided_3: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_24, [6, 128, 128], [16384, 128, 1])
    as_strided_26: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_24, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_7: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_24, as_strided_26, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_177: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(as_strided_12, as_strided_26);  as_strided_12 = as_strided_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_62: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_377, as_strided_scatter_7);  permute_377 = None
    bmm_63: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_7, permute_378);  as_strided_scatter_7 = permute_378 = None
    view_660: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_62, [1, 6, 64, 128]);  bmm_62 = None
    view_661: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_63, [1, 6, 128, 64]);  bmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_379: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_660, [0, 1, 3, 2]);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_178: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_27, permute_379);  tangents_27 = permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_380: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_176, [0, 2, 1, 3]);  add_176 = None
    clone_50: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
    view_662: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 384]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_663: "f32[128, 384]" = torch.ops.aten.reshape.default(view_662, [128, 384]);  view_662 = None
    permute_381: "f32[384, 128]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_185: "f32[384, 512]" = torch.ops.aten.mm.default(permute_381, view_486);  permute_381 = None
    permute_382: "f32[512, 384]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    mm_186: "f32[128, 512]" = torch.ops.aten.mm.default(view_663, permute_383);  view_663 = permute_383 = None
    view_664: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_186, [1, 128, 512]);  mm_186 = None
    permute_384: "f32[384, 512]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_385: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_178, [0, 2, 1, 3]);  add_178 = None
    clone_51: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_665: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_51, [1, 128, 384]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_666: "f32[128, 384]" = torch.ops.aten.reshape.default(view_665, [128, 384]);  view_665 = None
    permute_386: "f32[384, 128]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_187: "f32[384, 512]" = torch.ops.aten.mm.default(permute_386, view_486);  permute_386 = None
    permute_387: "f32[512, 384]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    mm_188: "f32[128, 512]" = torch.ops.aten.mm.default(view_666, permute_388);  view_666 = permute_388 = None
    view_667: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_188, [1, 128, 512]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_179: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_664, view_667);  view_664 = view_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_389: "f32[384, 512]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_390: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_661, [0, 2, 1, 3]);  view_661 = None
    clone_52: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
    view_668: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_52, [1, 128, 384]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_669: "f32[128, 384]" = torch.ops.aten.reshape.default(view_668, [128, 384]);  view_668 = None
    permute_391: "f32[384, 128]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_189: "f32[384, 512]" = torch.ops.aten.mm.default(permute_391, view_486);  permute_391 = view_486 = None
    permute_392: "f32[512, 384]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    mm_190: "f32[128, 512]" = torch.ops.aten.mm.default(view_669, permute_393);  view_669 = permute_393 = None
    view_670: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_190, [1, 128, 512]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_180: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_179, view_670);  add_179 = view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_394: "f32[384, 512]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_277: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_180, primals_36);  primals_36 = None
    mul_278: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_180, mul_147);  add_180 = mul_147 = None
    sum_45: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
    view_671: "f32[512]" = torch.ops.aten.reshape.default(sum_45, [512]);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_279: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_277, add_124)
    mul_280: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_277, rsqrt_35);  mul_277 = None
    sum_46: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_181: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_175, mul_280);  add_175 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_73: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_35, 3);  rsqrt_35 = None
    mul_281: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_46, -0.5);  sum_46 = None
    mul_282: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_281, pow_73);  mul_281 = pow_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_102: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_282, [1, 128, 512]);  mul_282 = None
    div_36: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
    pow_74: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 1.0);  add_124 = None
    mul_283: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
    mul_284: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_283);  div_36 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_182: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_181, mul_284);  add_181 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_21: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_285: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_286: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_182, mul_285);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_672: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_286, [128, 512]);  mul_286 = None
    permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(view_672, [1, 0])
    mm_191: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_395, view_484);  permute_395 = view_484 = None
    permute_396: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    mm_192: "f32[128, 1024]" = torch.ops.aten.mm.default(view_672, permute_397);  view_672 = permute_397 = None
    view_673: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_192, [1, 128, 1024]);  mm_192 = None
    permute_398: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None
    mul_287: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_288: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_673, mul_287);  view_673 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_289: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_288, mul_145);  mul_145 = None
    mul_290: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_288, view_483);  mul_288 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_674: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_289, [128, 1024]);  mul_289 = None
    permute_399: "f32[1024, 128]" = torch.ops.aten.permute.default(view_674, [1, 0])
    mm_193: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_399, view_480);  permute_399 = None
    permute_400: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    mm_194: "f32[128, 512]" = torch.ops.aten.mm.default(view_674, permute_401);  view_674 = permute_401 = None
    view_675: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_194, [1, 128, 512]);  mm_194 = None
    permute_402: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_291: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_290, mul_142);  mul_142 = None
    mul_292: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_290, add_123);  mul_290 = add_123 = None
    mul_293: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
    sub_38: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_293);  mul_293 = None
    mul_294: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_291, sub_38);  mul_291 = sub_38 = None
    mul_295: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_294, 0.7978845608028654);  mul_294 = None
    mul_296: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_295, 0.044715)
    pow_75: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 2.0);  view_481 = None
    mul_297: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_75, 3.0);  pow_75 = None
    mul_298: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_183: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_295, mul_298);  mul_295 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_299: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_292, 0.5);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_184: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_183, mul_299);  add_183 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_676: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_184, [128, 1024]);  add_184 = None
    permute_403: "f32[1024, 128]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_195: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_403, view_480);  permute_403 = view_480 = None
    permute_404: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    mm_196: "f32[128, 512]" = torch.ops.aten.mm.default(view_676, permute_405);  view_676 = permute_405 = None
    view_677: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_196, [1, 128, 512]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_185: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_675, view_677);  view_675 = view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_406: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_300: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_185, primals_35);  primals_35 = None
    mul_301: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_185, mul_140);  add_185 = mul_140 = None
    sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1], True);  mul_301 = None
    view_678: "f32[512]" = torch.ops.aten.reshape.default(sum_47, [512]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_302: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_300, add_120)
    mul_303: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_300, rsqrt_34);  mul_300 = None
    sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_186: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_182, mul_303);  add_182 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_76: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_34, 3);  rsqrt_34 = None
    mul_304: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
    mul_305: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_304, pow_76);  mul_304 = pow_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_103: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_305, [1, 128, 512]);  mul_305 = None
    div_37: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
    pow_77: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_120, 1.0);  add_120 = None
    mul_306: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_77, 2.0);  pow_77 = None
    mul_307: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_306);  div_37 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_187: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_186, mul_307);  add_186 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_23: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_308: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_309: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_187, mul_308);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_679: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_309, [128, 512]);  mul_309 = None
    permute_407: "f32[512, 128]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_197: "f32[512, 384]" = torch.ops.aten.mm.default(permute_407, view_478);  permute_407 = view_478 = None
    permute_408: "f32[384, 512]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    mm_198: "f32[128, 384]" = torch.ops.aten.mm.default(view_679, permute_409);  view_679 = permute_409 = None
    view_680: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_198, [1, 128, 384]);  mm_198 = None
    permute_410: "f32[512, 384]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_681: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_680, [1, 128, 6, 64]);  view_680 = None
    permute_411: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_682: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_411, [6, 128, 64]);  permute_411 = None
    bmm_64: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_412, view_682);  permute_412 = None
    bmm_65: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_682, permute_413);  view_682 = permute_413 = None
    view_683: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_64, [1, 6, 128, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_188: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_26, view_683);  tangents_26 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_684: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [1, 6, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_24: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
    mul_310: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_311: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_684, mul_310);  view_684 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_312: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_311, alias_99);  mul_311 = None
    sum_49: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [-1], True)
    mul_313: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_99, sum_49);  alias_99 = sum_49 = None
    sub_39: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_5: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_39, 0);  sub_39 = None
    as_strided_scatter_8: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_5, [6, 128, 128], [16384, 128, 1], 0);  squeeze_5 = None
    as_strided_31: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_4: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_31, [6, 128, 128], [16384, 128, 1])
    as_strided_33: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_31, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_9: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_31, as_strided_33, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_31 = as_strided_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_66: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_414, as_strided_scatter_9);  permute_414 = None
    bmm_67: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_9, permute_415);  as_strided_scatter_9 = permute_415 = None
    view_685: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_66, [1, 6, 64, 128]);  bmm_66 = None
    view_686: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_67, [1, 6, 128, 64]);  bmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_416: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_685, [0, 1, 3, 2]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_189: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_25, permute_416);  tangents_25 = permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_417: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_188, [0, 2, 1, 3]);  add_188 = None
    clone_58: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
    view_687: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 384]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_688: "f32[128, 384]" = torch.ops.aten.reshape.default(view_687, [128, 384]);  view_687 = None
    permute_418: "f32[384, 128]" = torch.ops.aten.permute.default(view_688, [1, 0])
    mm_199: "f32[384, 512]" = torch.ops.aten.mm.default(permute_418, view_233);  permute_418 = None
    permute_419: "f32[512, 384]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    mm_200: "f32[128, 512]" = torch.ops.aten.mm.default(view_688, permute_420);  view_688 = permute_420 = None
    view_689: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_200, [1, 128, 512]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_190: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_173, view_689);  add_173 = view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_421: "f32[384, 512]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_422: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_189, [0, 2, 1, 3]);  add_189 = None
    clone_59: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
    view_690: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_59, [1, 128, 384]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_691: "f32[128, 384]" = torch.ops.aten.reshape.default(view_690, [128, 384]);  view_690 = None
    permute_423: "f32[384, 128]" = torch.ops.aten.permute.default(view_691, [1, 0])
    mm_201: "f32[384, 512]" = torch.ops.aten.mm.default(permute_423, view_233);  permute_423 = None
    permute_424: "f32[512, 384]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    mm_202: "f32[128, 512]" = torch.ops.aten.mm.default(view_691, permute_425);  view_691 = permute_425 = None
    view_692: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_202, [1, 128, 512]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_191: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_190, view_692);  add_190 = view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_426: "f32[384, 512]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_427: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
    clone_60: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    view_693: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_60, [1, 128, 384]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_694: "f32[128, 384]" = torch.ops.aten.reshape.default(view_693, [128, 384]);  view_693 = None
    permute_428: "f32[384, 128]" = torch.ops.aten.permute.default(view_694, [1, 0])
    mm_203: "f32[384, 512]" = torch.ops.aten.mm.default(permute_428, view_460);  permute_428 = view_460 = None
    permute_429: "f32[512, 384]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    mm_204: "f32[128, 512]" = torch.ops.aten.mm.default(view_694, permute_430);  view_694 = permute_430 = None
    view_695: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_204, [1, 128, 512]);  mm_204 = None
    permute_431: "f32[384, 512]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_314: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_695, primals_34);  primals_34 = None
    mul_315: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_695, mul_138);  view_695 = mul_138 = None
    sum_50: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1], True);  mul_315 = None
    view_696: "f32[512]" = torch.ops.aten.reshape.default(sum_50, [512]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_316: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_314, add_117)
    mul_317: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_314, rsqrt_33);  mul_314 = None
    sum_51: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_192: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_187, mul_317);  add_187 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_78: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_33, 3);  rsqrt_33 = None
    mul_318: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_51, -0.5);  sum_51 = None
    mul_319: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_318, pow_78);  mul_318 = pow_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_104: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_319, [1, 128, 512]);  mul_319 = None
    div_38: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_104, 512);  expand_104 = None
    pow_79: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_117, 1.0);  add_117 = None
    mul_320: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_79, 2.0);  pow_79 = None
    mul_321: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_320);  div_38 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_193: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_192, mul_321);  add_192 = mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_25: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None
    mul_322: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_323: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_193, mul_322);  mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_697: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_323, [128, 512]);  mul_323 = None
    permute_432: "f32[512, 128]" = torch.ops.aten.permute.default(view_697, [1, 0])
    mm_205: "f32[512, 384]" = torch.ops.aten.mm.default(permute_432, view_458);  permute_432 = view_458 = None
    permute_433: "f32[384, 512]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    mm_206: "f32[128, 384]" = torch.ops.aten.mm.default(view_697, permute_434);  view_697 = permute_434 = None
    view_698: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_206, [1, 128, 384]);  mm_206 = None
    permute_435: "f32[512, 384]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_699: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_698, [1, 128, 6, 64]);  view_698 = None
    permute_436: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_700: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_436, [6, 128, 64]);  permute_436 = None
    bmm_68: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_437, view_700);  permute_437 = None
    bmm_69: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_700, permute_438);  view_700 = permute_438 = None
    view_701: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_68, [1, 6, 128, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_194: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_701);  tangents_24 = view_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_702: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [1, 6, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_26: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_324: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_325: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_702, mul_324);  view_702 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_326: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_325, alias_101);  mul_325 = None
    sum_52: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
    mul_327: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_101, sum_52);  alias_101 = sum_52 = None
    sub_40: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_6: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_40, 0);  sub_40 = None
    as_strided_scatter_10: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_6, [6, 128, 128], [16384, 128, 1], 0);  squeeze_6 = None
    as_strided_38: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_5: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_38, [6, 128, 128], [16384, 128, 1])
    as_strided_40: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_38, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_11: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_38, as_strided_40, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_195: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_177, as_strided_40);  add_177 = as_strided_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_70: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_439, as_strided_scatter_11);  permute_439 = None
    bmm_71: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_11, permute_440);  as_strided_scatter_11 = permute_440 = None
    view_703: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_70, [1, 6, 64, 128]);  bmm_70 = None
    view_704: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_71, [1, 6, 128, 64]);  bmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_441: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_703, [0, 1, 3, 2]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_196: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_23, permute_441);  tangents_23 = permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_442: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_194, [0, 2, 1, 3]);  add_194 = None
    clone_64: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
    view_705: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_64, [1, 128, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_706: "f32[128, 384]" = torch.ops.aten.reshape.default(view_705, [128, 384]);  view_705 = None
    permute_443: "f32[384, 128]" = torch.ops.aten.permute.default(view_706, [1, 0])
    mm_207: "f32[384, 512]" = torch.ops.aten.mm.default(permute_443, view_440);  permute_443 = None
    permute_444: "f32[512, 384]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    mm_208: "f32[128, 512]" = torch.ops.aten.mm.default(view_706, permute_445);  view_706 = permute_445 = None
    view_707: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_208, [1, 128, 512]);  mm_208 = None
    permute_446: "f32[384, 512]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_447: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_196, [0, 2, 1, 3]);  add_196 = None
    clone_65: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_708: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 384]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_709: "f32[128, 384]" = torch.ops.aten.reshape.default(view_708, [128, 384]);  view_708 = None
    permute_448: "f32[384, 128]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_209: "f32[384, 512]" = torch.ops.aten.mm.default(permute_448, view_440);  permute_448 = None
    permute_449: "f32[512, 384]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    mm_210: "f32[128, 512]" = torch.ops.aten.mm.default(view_709, permute_450);  view_709 = permute_450 = None
    view_710: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_210, [1, 128, 512]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_197: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_707, view_710);  view_707 = view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_451: "f32[384, 512]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_452: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    clone_66: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_711: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 384]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_712: "f32[128, 384]" = torch.ops.aten.reshape.default(view_711, [128, 384]);  view_711 = None
    permute_453: "f32[384, 128]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_211: "f32[384, 512]" = torch.ops.aten.mm.default(permute_453, view_440);  permute_453 = view_440 = None
    permute_454: "f32[512, 384]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    mm_212: "f32[128, 512]" = torch.ops.aten.mm.default(view_712, permute_455);  view_712 = permute_455 = None
    view_713: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_212, [1, 128, 512]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_198: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_197, view_713);  add_197 = view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_456: "f32[384, 512]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_328: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_198, primals_33);  primals_33 = None
    mul_329: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_198, mul_136);  add_198 = mul_136 = None
    sum_53: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1], True);  mul_329 = None
    view_714: "f32[512]" = torch.ops.aten.reshape.default(sum_53, [512]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_330: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_328, add_114)
    mul_331: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_328, rsqrt_32);  mul_328 = None
    sum_54: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_199: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_193, mul_331);  add_193 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_80: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_32, 3);  rsqrt_32 = None
    mul_332: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
    mul_333: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_332, pow_80);  mul_332 = pow_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_105: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_333, [1, 128, 512]);  mul_333 = None
    div_39: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_105, 512);  expand_105 = None
    pow_81: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_114, 1.0);  add_114 = None
    mul_334: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_81, 2.0);  pow_81 = None
    mul_335: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_334);  div_39 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_200: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_199, mul_335);  add_199 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_27: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
    mul_336: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_337: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_200, mul_336);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_715: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_337, [128, 512]);  mul_337 = None
    permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_213: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_457, view_438);  permute_457 = view_438 = None
    permute_458: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    mm_214: "f32[128, 1024]" = torch.ops.aten.mm.default(view_715, permute_459);  view_715 = permute_459 = None
    view_716: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_214, [1, 128, 1024]);  mm_214 = None
    permute_460: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_338: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_339: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_716, mul_338);  view_716 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_340: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_339, mul_134);  mul_134 = None
    mul_341: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_339, view_437);  mul_339 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_717: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_340, [128, 1024]);  mul_340 = None
    permute_461: "f32[1024, 128]" = torch.ops.aten.permute.default(view_717, [1, 0])
    mm_215: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_461, view_434);  permute_461 = None
    permute_462: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    mm_216: "f32[128, 512]" = torch.ops.aten.mm.default(view_717, permute_463);  view_717 = permute_463 = None
    view_718: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_216, [1, 128, 512]);  mm_216 = None
    permute_464: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_342: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_341, mul_131);  mul_131 = None
    mul_343: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_341, add_113);  mul_341 = add_113 = None
    mul_344: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
    sub_41: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_344);  mul_344 = None
    mul_345: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_342, sub_41);  mul_342 = sub_41 = None
    mul_346: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_345, 0.7978845608028654);  mul_345 = None
    mul_347: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_346, 0.044715)
    pow_82: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_435, 2.0);  view_435 = None
    mul_348: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_82, 3.0);  pow_82 = None
    mul_349: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_201: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_346, mul_349);  mul_346 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_350: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_343, 0.5);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_202: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_201, mul_350);  add_201 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_719: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_202, [128, 1024]);  add_202 = None
    permute_465: "f32[1024, 128]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_217: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_465, view_434);  permute_465 = view_434 = None
    permute_466: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    mm_218: "f32[128, 512]" = torch.ops.aten.mm.default(view_719, permute_467);  view_719 = permute_467 = None
    view_720: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_218, [1, 128, 512]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_203: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_718, view_720);  view_718 = view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_468: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_351: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_203, primals_32);  primals_32 = None
    mul_352: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_203, mul_129);  add_203 = mul_129 = None
    sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1], True);  mul_352 = None
    view_721: "f32[512]" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_353: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_351, add_110)
    mul_354: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_351, rsqrt_31);  mul_351 = None
    sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_204: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_200, mul_354);  add_200 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_83: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
    mul_355: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
    mul_356: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_355, pow_83);  mul_355 = pow_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_106: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_356, [1, 128, 512]);  mul_356 = None
    div_40: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_106, 512);  expand_106 = None
    pow_84: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_110, 1.0);  add_110 = None
    mul_357: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
    mul_358: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_357);  div_40 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_205: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_204, mul_358);  add_204 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_29: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_359: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_360: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_205, mul_359);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_722: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_360, [128, 512]);  mul_360 = None
    permute_469: "f32[512, 128]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_219: "f32[512, 384]" = torch.ops.aten.mm.default(permute_469, view_432);  permute_469 = view_432 = None
    permute_470: "f32[384, 512]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    mm_220: "f32[128, 384]" = torch.ops.aten.mm.default(view_722, permute_471);  view_722 = permute_471 = None
    view_723: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_220, [1, 128, 384]);  mm_220 = None
    permute_472: "f32[512, 384]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_724: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_723, [1, 128, 6, 64]);  view_723 = None
    permute_473: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_725: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_473, [6, 128, 64]);  permute_473 = None
    bmm_72: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_474, view_725);  permute_474 = None
    bmm_73: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_725, permute_475);  view_725 = permute_475 = None
    view_726: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_72, [1, 6, 128, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_206: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_726);  tangents_22 = view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_727: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_73, [1, 6, 128, 128]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_30: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
    mul_361: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_362: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_727, mul_361);  view_727 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_363: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_362, alias_105);  mul_362 = None
    sum_57: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
    mul_364: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_105, sum_57);  alias_105 = sum_57 = None
    sub_42: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_7: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_42, 0);  sub_42 = None
    as_strided_scatter_12: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_7, [6, 128, 128], [16384, 128, 1], 0);  squeeze_7 = None
    as_strided_45: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_12, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_12 = None
    new_empty_strided_6: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_45, [6, 128, 128], [16384, 128, 1])
    as_strided_47: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_45, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_13: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_45, as_strided_47, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_45 = as_strided_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_74: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_476, as_strided_scatter_13);  permute_476 = None
    bmm_75: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_13, permute_477);  as_strided_scatter_13 = permute_477 = None
    view_728: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_74, [1, 6, 64, 128]);  bmm_74 = None
    view_729: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_75, [1, 6, 128, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_478: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_728, [0, 1, 3, 2]);  view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_207: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_21, permute_478);  tangents_21 = permute_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_479: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_206, [0, 2, 1, 3]);  add_206 = None
    clone_72: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_730: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_72, [1, 128, 384]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_731: "f32[128, 384]" = torch.ops.aten.reshape.default(view_730, [128, 384]);  view_730 = None
    permute_480: "f32[384, 128]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_221: "f32[384, 512]" = torch.ops.aten.mm.default(permute_480, view_233);  permute_480 = None
    permute_481: "f32[512, 384]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    mm_222: "f32[128, 512]" = torch.ops.aten.mm.default(view_731, permute_482);  view_731 = permute_482 = None
    view_732: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_222, [1, 128, 512]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_208: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_191, view_732);  add_191 = view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_483: "f32[384, 512]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_484: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_207, [0, 2, 1, 3]);  add_207 = None
    clone_73: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    view_733: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 384]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_734: "f32[128, 384]" = torch.ops.aten.reshape.default(view_733, [128, 384]);  view_733 = None
    permute_485: "f32[384, 128]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_223: "f32[384, 512]" = torch.ops.aten.mm.default(permute_485, view_233);  permute_485 = None
    permute_486: "f32[512, 384]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    mm_224: "f32[128, 512]" = torch.ops.aten.mm.default(view_734, permute_487);  view_734 = permute_487 = None
    view_735: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_224, [1, 128, 512]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_209: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_208, view_735);  add_208 = view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_488: "f32[384, 512]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_489: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_729, [0, 2, 1, 3]);  view_729 = None
    clone_74: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_736: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 384]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_737: "f32[128, 384]" = torch.ops.aten.reshape.default(view_736, [128, 384]);  view_736 = None
    permute_490: "f32[384, 128]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_225: "f32[384, 512]" = torch.ops.aten.mm.default(permute_490, view_414);  permute_490 = view_414 = None
    permute_491: "f32[512, 384]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    mm_226: "f32[128, 512]" = torch.ops.aten.mm.default(view_737, permute_492);  view_737 = permute_492 = None
    view_738: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_226, [1, 128, 512]);  mm_226 = None
    permute_493: "f32[384, 512]" = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_365: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_738, primals_31);  primals_31 = None
    mul_366: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_738, mul_127);  view_738 = mul_127 = None
    sum_58: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1], True);  mul_366 = None
    view_739: "f32[512]" = torch.ops.aten.reshape.default(sum_58, [512]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_367: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_365, add_107)
    mul_368: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_365, rsqrt_30);  mul_365 = None
    sum_59: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_210: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_205, mul_368);  add_205 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_85: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
    mul_369: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_59, -0.5);  sum_59 = None
    mul_370: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_369, pow_85);  mul_369 = pow_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_107: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_370, [1, 128, 512]);  mul_370 = None
    div_41: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_107, 512);  expand_107 = None
    pow_86: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_107, 1.0);  add_107 = None
    mul_371: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
    mul_372: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_371);  div_41 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_211: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_210, mul_372);  add_210 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_31: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_373: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_374: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, mul_373);  mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_740: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_374, [128, 512]);  mul_374 = None
    permute_494: "f32[512, 128]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_227: "f32[512, 384]" = torch.ops.aten.mm.default(permute_494, view_412);  permute_494 = view_412 = None
    permute_495: "f32[384, 512]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    mm_228: "f32[128, 384]" = torch.ops.aten.mm.default(view_740, permute_496);  view_740 = permute_496 = None
    view_741: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_228, [1, 128, 384]);  mm_228 = None
    permute_497: "f32[512, 384]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_742: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_741, [1, 128, 6, 64]);  view_741 = None
    permute_498: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_743: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_498, [6, 128, 64]);  permute_498 = None
    bmm_76: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_499, view_743);  permute_499 = None
    bmm_77: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_743, permute_500);  view_743 = permute_500 = None
    view_744: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_76, [1, 6, 128, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_212: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_744);  tangents_20 = view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_745: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_77, [1, 6, 128, 128]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_32: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
    mul_375: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_376: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_745, mul_375);  view_745 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_377: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_376, alias_107);  mul_376 = None
    sum_60: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
    mul_378: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_107, sum_60);  alias_107 = sum_60 = None
    sub_43: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_8: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_43, 0);  sub_43 = None
    as_strided_scatter_14: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_8, [6, 128, 128], [16384, 128, 1], 0);  squeeze_8 = None
    as_strided_52: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_14, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_14 = None
    new_empty_strided_7: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_52, [6, 128, 128], [16384, 128, 1])
    as_strided_54: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_52, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_15: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_52, as_strided_54, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_213: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_195, as_strided_54);  add_195 = as_strided_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_78: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_501, as_strided_scatter_15);  permute_501 = None
    bmm_79: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_15, permute_502);  as_strided_scatter_15 = permute_502 = None
    view_746: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_78, [1, 6, 64, 128]);  bmm_78 = None
    view_747: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_79, [1, 6, 128, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_503: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_746, [0, 1, 3, 2]);  view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_214: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_19, permute_503);  tangents_19 = permute_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_504: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_212, [0, 2, 1, 3]);  add_212 = None
    clone_78: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
    view_748: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 384]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_749: "f32[128, 384]" = torch.ops.aten.reshape.default(view_748, [128, 384]);  view_748 = None
    permute_505: "f32[384, 128]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_229: "f32[384, 512]" = torch.ops.aten.mm.default(permute_505, view_394);  permute_505 = None
    permute_506: "f32[512, 384]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    mm_230: "f32[128, 512]" = torch.ops.aten.mm.default(view_749, permute_507);  view_749 = permute_507 = None
    view_750: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_230, [1, 128, 512]);  mm_230 = None
    permute_508: "f32[384, 512]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_509: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_214, [0, 2, 1, 3]);  add_214 = None
    clone_79: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
    view_751: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_79, [1, 128, 384]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_752: "f32[128, 384]" = torch.ops.aten.reshape.default(view_751, [128, 384]);  view_751 = None
    permute_510: "f32[384, 128]" = torch.ops.aten.permute.default(view_752, [1, 0])
    mm_231: "f32[384, 512]" = torch.ops.aten.mm.default(permute_510, view_394);  permute_510 = None
    permute_511: "f32[512, 384]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    mm_232: "f32[128, 512]" = torch.ops.aten.mm.default(view_752, permute_512);  view_752 = permute_512 = None
    view_753: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_232, [1, 128, 512]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_215: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_750, view_753);  view_750 = view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_513: "f32[384, 512]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_514: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_747, [0, 2, 1, 3]);  view_747 = None
    clone_80: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_514, memory_format = torch.contiguous_format);  permute_514 = None
    view_754: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_80, [1, 128, 384]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_755: "f32[128, 384]" = torch.ops.aten.reshape.default(view_754, [128, 384]);  view_754 = None
    permute_515: "f32[384, 128]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_233: "f32[384, 512]" = torch.ops.aten.mm.default(permute_515, view_394);  permute_515 = view_394 = None
    permute_516: "f32[512, 384]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    mm_234: "f32[128, 512]" = torch.ops.aten.mm.default(view_755, permute_517);  view_755 = permute_517 = None
    view_756: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_234, [1, 128, 512]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_216: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_215, view_756);  add_215 = view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_518: "f32[384, 512]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_379: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_216, primals_30);  primals_30 = None
    mul_380: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_216, mul_125);  add_216 = mul_125 = None
    sum_61: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1], True);  mul_380 = None
    view_757: "f32[512]" = torch.ops.aten.reshape.default(sum_61, [512]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_381: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_379, add_104)
    mul_382: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_379, rsqrt_29);  mul_379 = None
    sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_217: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_211, mul_382);  add_211 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_87: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
    mul_383: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
    mul_384: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_383, pow_87);  mul_383 = pow_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_108: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_384, [1, 128, 512]);  mul_384 = None
    div_42: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_108, 512);  expand_108 = None
    pow_88: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 1.0);  add_104 = None
    mul_385: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
    mul_386: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_385);  div_42 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_218: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_217, mul_386);  add_217 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_33: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_387: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_388: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_218, mul_387);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_758: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_388, [128, 512]);  mul_388 = None
    permute_519: "f32[512, 128]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_235: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_519, view_392);  permute_519 = view_392 = None
    permute_520: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    mm_236: "f32[128, 1024]" = torch.ops.aten.mm.default(view_758, permute_521);  view_758 = permute_521 = None
    view_759: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_236, [1, 128, 1024]);  mm_236 = None
    permute_522: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_389: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_390: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_759, mul_389);  view_759 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_391: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_390, mul_123);  mul_123 = None
    mul_392: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_390, view_391);  mul_390 = view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_760: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_391, [128, 1024]);  mul_391 = None
    permute_523: "f32[1024, 128]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_237: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_523, view_388);  permute_523 = None
    permute_524: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    mm_238: "f32[128, 512]" = torch.ops.aten.mm.default(view_760, permute_525);  view_760 = permute_525 = None
    view_761: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_238, [1, 128, 512]);  mm_238 = None
    permute_526: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_393: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_392, mul_120);  mul_120 = None
    mul_394: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_392, add_103);  mul_392 = add_103 = None
    mul_395: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
    sub_44: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_395);  mul_395 = None
    mul_396: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_393, sub_44);  mul_393 = sub_44 = None
    mul_397: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_396, 0.7978845608028654);  mul_396 = None
    mul_398: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_397, 0.044715)
    pow_89: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 2.0);  view_389 = None
    mul_399: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_89, 3.0);  pow_89 = None
    mul_400: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_219: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_397, mul_400);  mul_397 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_401: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_394, 0.5);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_220: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_219, mul_401);  add_219 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_762: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_220, [128, 1024]);  add_220 = None
    permute_527: "f32[1024, 128]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_239: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_527, view_388);  permute_527 = view_388 = None
    permute_528: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    mm_240: "f32[128, 512]" = torch.ops.aten.mm.default(view_762, permute_529);  view_762 = permute_529 = None
    view_763: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_240, [1, 128, 512]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_221: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_761, view_763);  view_761 = view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_530: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_402: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_221, primals_29);  primals_29 = None
    mul_403: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_221, mul_118);  add_221 = mul_118 = None
    sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1], True);  mul_403 = None
    view_764: "f32[512]" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_404: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_402, add_100)
    mul_405: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_402, rsqrt_28);  mul_402 = None
    sum_64: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_222: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_218, mul_405);  add_218 = mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_90: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
    mul_406: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
    mul_407: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_406, pow_90);  mul_406 = pow_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_109: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_407, [1, 128, 512]);  mul_407 = None
    div_43: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_109, 512);  expand_109 = None
    pow_91: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_100, 1.0);  add_100 = None
    mul_408: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_91, 2.0);  pow_91 = None
    mul_409: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_408);  div_43 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_223: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_222, mul_409);  add_222 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_35: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
    mul_410: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_411: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_223, mul_410);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_765: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_411, [128, 512]);  mul_411 = None
    permute_531: "f32[512, 128]" = torch.ops.aten.permute.default(view_765, [1, 0])
    mm_241: "f32[512, 384]" = torch.ops.aten.mm.default(permute_531, view_386);  permute_531 = view_386 = None
    permute_532: "f32[384, 512]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    mm_242: "f32[128, 384]" = torch.ops.aten.mm.default(view_765, permute_533);  view_765 = permute_533 = None
    view_766: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_242, [1, 128, 384]);  mm_242 = None
    permute_534: "f32[512, 384]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_767: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_766, [1, 128, 6, 64]);  view_766 = None
    permute_535: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_767, [0, 2, 1, 3]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_768: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_535, [6, 128, 64]);  permute_535 = None
    bmm_80: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_536, view_768);  permute_536 = None
    bmm_81: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_768, permute_537);  view_768 = permute_537 = None
    view_769: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_80, [1, 6, 128, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_224: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_769);  tangents_18 = view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_770: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_81, [1, 6, 128, 128]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_36: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_412: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_413: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_770, mul_412);  view_770 = mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_414: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_413, alias_111);  mul_413 = None
    sum_65: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [-1], True)
    mul_415: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_111, sum_65);  alias_111 = sum_65 = None
    sub_45: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_9: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_45, 0);  sub_45 = None
    as_strided_scatter_16: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_9, [6, 128, 128], [16384, 128, 1], 0);  squeeze_9 = None
    as_strided_59: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_16, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_16 = None
    new_empty_strided_8: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_59, [6, 128, 128], [16384, 128, 1])
    as_strided_61: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_59, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_17: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_59, as_strided_61, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_59 = as_strided_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_82: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_538, as_strided_scatter_17);  permute_538 = None
    bmm_83: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_17, permute_539);  as_strided_scatter_17 = permute_539 = None
    view_771: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_82, [1, 6, 64, 128]);  bmm_82 = None
    view_772: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_83, [1, 6, 128, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_540: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_771, [0, 1, 3, 2]);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_225: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_17, permute_540);  tangents_17 = permute_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_541: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_224, [0, 2, 1, 3]);  add_224 = None
    clone_86: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
    view_773: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 384]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_774: "f32[128, 384]" = torch.ops.aten.reshape.default(view_773, [128, 384]);  view_773 = None
    permute_542: "f32[384, 128]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_243: "f32[384, 512]" = torch.ops.aten.mm.default(permute_542, view_233);  permute_542 = None
    permute_543: "f32[512, 384]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    mm_244: "f32[128, 512]" = torch.ops.aten.mm.default(view_774, permute_544);  view_774 = permute_544 = None
    view_775: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_244, [1, 128, 512]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_226: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_209, view_775);  add_209 = view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_545: "f32[384, 512]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_546: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_225, [0, 2, 1, 3]);  add_225 = None
    clone_87: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
    view_776: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_87, [1, 128, 384]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_777: "f32[128, 384]" = torch.ops.aten.reshape.default(view_776, [128, 384]);  view_776 = None
    permute_547: "f32[384, 128]" = torch.ops.aten.permute.default(view_777, [1, 0])
    mm_245: "f32[384, 512]" = torch.ops.aten.mm.default(permute_547, view_233);  permute_547 = None
    permute_548: "f32[512, 384]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    mm_246: "f32[128, 512]" = torch.ops.aten.mm.default(view_777, permute_549);  view_777 = permute_549 = None
    view_778: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_246, [1, 128, 512]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_227: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_226, view_778);  add_226 = view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_550: "f32[384, 512]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_551: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    clone_88: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_779: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_88, [1, 128, 384]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_780: "f32[128, 384]" = torch.ops.aten.reshape.default(view_779, [128, 384]);  view_779 = None
    permute_552: "f32[384, 128]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_247: "f32[384, 512]" = torch.ops.aten.mm.default(permute_552, view_368);  permute_552 = view_368 = None
    permute_553: "f32[512, 384]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    mm_248: "f32[128, 512]" = torch.ops.aten.mm.default(view_780, permute_554);  view_780 = permute_554 = None
    view_781: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_248, [1, 128, 512]);  mm_248 = None
    permute_555: "f32[384, 512]" = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_416: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_781, primals_28);  primals_28 = None
    mul_417: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_781, mul_116);  view_781 = mul_116 = None
    sum_66: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1], True);  mul_417 = None
    view_782: "f32[512]" = torch.ops.aten.reshape.default(sum_66, [512]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_418: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_416, add_97)
    mul_419: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_416, rsqrt_27);  mul_416 = None
    sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_228: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_223, mul_419);  add_223 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_92: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
    mul_420: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_67, -0.5);  sum_67 = None
    mul_421: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_420, pow_92);  mul_420 = pow_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_110: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_421, [1, 128, 512]);  mul_421 = None
    div_44: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_110, 512);  expand_110 = None
    pow_93: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_97, 1.0);  add_97 = None
    mul_422: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_93, 2.0);  pow_93 = None
    mul_423: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_422);  div_44 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_229: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_228, mul_423);  add_228 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_37: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
    mul_424: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_425: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_229, mul_424);  mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_783: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_425, [128, 512]);  mul_425 = None
    permute_556: "f32[512, 128]" = torch.ops.aten.permute.default(view_783, [1, 0])
    mm_249: "f32[512, 384]" = torch.ops.aten.mm.default(permute_556, view_366);  permute_556 = view_366 = None
    permute_557: "f32[384, 512]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    mm_250: "f32[128, 384]" = torch.ops.aten.mm.default(view_783, permute_558);  view_783 = permute_558 = None
    view_784: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_250, [1, 128, 384]);  mm_250 = None
    permute_559: "f32[512, 384]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_785: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_784, [1, 128, 6, 64]);  view_784 = None
    permute_560: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_785, [0, 2, 1, 3]);  view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_786: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_560, [6, 128, 64]);  permute_560 = None
    bmm_84: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_561, view_786);  permute_561 = None
    bmm_85: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_786, permute_562);  view_786 = permute_562 = None
    view_787: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_84, [1, 6, 128, 64]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_230: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_787);  tangents_16 = view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_788: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_85, [1, 6, 128, 128]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_38: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_426: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_427: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_788, mul_426);  view_788 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_428: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_427, alias_113);  mul_427 = None
    sum_68: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
    mul_429: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_113, sum_68);  alias_113 = sum_68 = None
    sub_46: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_10: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_46, 0);  sub_46 = None
    as_strided_scatter_18: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_10, [6, 128, 128], [16384, 128, 1], 0);  squeeze_10 = None
    as_strided_66: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_9: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_66, [6, 128, 128], [16384, 128, 1])
    as_strided_68: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_66, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_19: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_66, as_strided_68, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_231: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_213, as_strided_68);  add_213 = as_strided_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_86: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_563, as_strided_scatter_19);  permute_563 = None
    bmm_87: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_19, permute_564);  as_strided_scatter_19 = permute_564 = None
    view_789: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_86, [1, 6, 64, 128]);  bmm_86 = None
    view_790: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_87, [1, 6, 128, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_565: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_789, [0, 1, 3, 2]);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_232: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_15, permute_565);  tangents_15 = permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_566: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_230, [0, 2, 1, 3]);  add_230 = None
    clone_92: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
    view_791: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_92, [1, 128, 384]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_792: "f32[128, 384]" = torch.ops.aten.reshape.default(view_791, [128, 384]);  view_791 = None
    permute_567: "f32[384, 128]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_251: "f32[384, 512]" = torch.ops.aten.mm.default(permute_567, view_348);  permute_567 = None
    permute_568: "f32[512, 384]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    mm_252: "f32[128, 512]" = torch.ops.aten.mm.default(view_792, permute_569);  view_792 = permute_569 = None
    view_793: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_252, [1, 128, 512]);  mm_252 = None
    permute_570: "f32[384, 512]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_571: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_232, [0, 2, 1, 3]);  add_232 = None
    clone_93: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
    view_794: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_93, [1, 128, 384]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_795: "f32[128, 384]" = torch.ops.aten.reshape.default(view_794, [128, 384]);  view_794 = None
    permute_572: "f32[384, 128]" = torch.ops.aten.permute.default(view_795, [1, 0])
    mm_253: "f32[384, 512]" = torch.ops.aten.mm.default(permute_572, view_348);  permute_572 = None
    permute_573: "f32[512, 384]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    mm_254: "f32[128, 512]" = torch.ops.aten.mm.default(view_795, permute_574);  view_795 = permute_574 = None
    view_796: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_254, [1, 128, 512]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_233: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_793, view_796);  view_793 = view_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_575: "f32[384, 512]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_576: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_790, [0, 2, 1, 3]);  view_790 = None
    clone_94: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_576, memory_format = torch.contiguous_format);  permute_576 = None
    view_797: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 384]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_798: "f32[128, 384]" = torch.ops.aten.reshape.default(view_797, [128, 384]);  view_797 = None
    permute_577: "f32[384, 128]" = torch.ops.aten.permute.default(view_798, [1, 0])
    mm_255: "f32[384, 512]" = torch.ops.aten.mm.default(permute_577, view_348);  permute_577 = view_348 = None
    permute_578: "f32[512, 384]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    mm_256: "f32[128, 512]" = torch.ops.aten.mm.default(view_798, permute_579);  view_798 = permute_579 = None
    view_799: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_256, [1, 128, 512]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_234: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_233, view_799);  add_233 = view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_580: "f32[384, 512]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_430: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_234, primals_27);  primals_27 = None
    mul_431: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_234, mul_114);  add_234 = mul_114 = None
    sum_69: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1], True);  mul_431 = None
    view_800: "f32[512]" = torch.ops.aten.reshape.default(sum_69, [512]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_432: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_430, add_94)
    mul_433: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_430, rsqrt_26);  mul_430 = None
    sum_70: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_235: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_229, mul_433);  add_229 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_94: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
    mul_434: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_70, -0.5);  sum_70 = None
    mul_435: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_434, pow_94);  mul_434 = pow_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_111: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_435, [1, 128, 512]);  mul_435 = None
    div_45: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_111, 512);  expand_111 = None
    pow_95: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_94, 1.0);  add_94 = None
    mul_436: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_95, 2.0);  pow_95 = None
    mul_437: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_436);  div_45 = mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_236: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_235, mul_437);  add_235 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_39: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_438: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_439: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_236, mul_438);  mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_801: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_439, [128, 512]);  mul_439 = None
    permute_581: "f32[512, 128]" = torch.ops.aten.permute.default(view_801, [1, 0])
    mm_257: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_581, view_346);  permute_581 = view_346 = None
    permute_582: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    mm_258: "f32[128, 1024]" = torch.ops.aten.mm.default(view_801, permute_583);  view_801 = permute_583 = None
    view_802: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_258, [1, 128, 1024]);  mm_258 = None
    permute_584: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
    mul_440: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_441: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_802, mul_440);  view_802 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_442: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_441, mul_112);  mul_112 = None
    mul_443: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_441, view_345);  mul_441 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_803: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_442, [128, 1024]);  mul_442 = None
    permute_585: "f32[1024, 128]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_259: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_585, view_342);  permute_585 = None
    permute_586: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    mm_260: "f32[128, 512]" = torch.ops.aten.mm.default(view_803, permute_587);  view_803 = permute_587 = None
    view_804: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_260, [1, 128, 512]);  mm_260 = None
    permute_588: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_444: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_443, mul_109);  mul_109 = None
    mul_445: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_443, add_93);  mul_443 = add_93 = None
    mul_446: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
    sub_47: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_446);  mul_446 = None
    mul_447: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_444, sub_47);  mul_444 = sub_47 = None
    mul_448: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_447, 0.7978845608028654);  mul_447 = None
    mul_449: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_448, 0.044715)
    pow_96: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_343, 2.0);  view_343 = None
    mul_450: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_96, 3.0);  pow_96 = None
    mul_451: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_237: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_448, mul_451);  mul_448 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_452: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_445, 0.5);  mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_238: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_452);  add_237 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_805: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_238, [128, 1024]);  add_238 = None
    permute_589: "f32[1024, 128]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_261: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_589, view_342);  permute_589 = view_342 = None
    permute_590: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    mm_262: "f32[128, 512]" = torch.ops.aten.mm.default(view_805, permute_591);  view_805 = permute_591 = None
    view_806: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_262, [1, 128, 512]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_239: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_804, view_806);  view_804 = view_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_592: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_590, [1, 0]);  permute_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_453: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_239, primals_26);  primals_26 = None
    mul_454: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_239, mul_107);  add_239 = mul_107 = None
    sum_71: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1], True);  mul_454 = None
    view_807: "f32[512]" = torch.ops.aten.reshape.default(sum_71, [512]);  sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_455: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_453, add_90)
    mul_456: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_453, rsqrt_25);  mul_453 = None
    sum_72: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_240: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_236, mul_456);  add_236 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_97: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
    mul_457: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
    mul_458: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_457, pow_97);  mul_457 = pow_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_112: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_458, [1, 128, 512]);  mul_458 = None
    div_46: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_112, 512);  expand_112 = None
    pow_98: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_90, 1.0);  add_90 = None
    mul_459: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_98, 2.0);  pow_98 = None
    mul_460: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_459);  div_46 = mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_241: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_240, mul_460);  add_240 = mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_41: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_461: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_462: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, mul_461);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_808: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_462, [128, 512]);  mul_462 = None
    permute_593: "f32[512, 128]" = torch.ops.aten.permute.default(view_808, [1, 0])
    mm_263: "f32[512, 384]" = torch.ops.aten.mm.default(permute_593, view_340);  permute_593 = view_340 = None
    permute_594: "f32[384, 512]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    mm_264: "f32[128, 384]" = torch.ops.aten.mm.default(view_808, permute_595);  view_808 = permute_595 = None
    view_809: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_264, [1, 128, 384]);  mm_264 = None
    permute_596: "f32[512, 384]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_810: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_809, [1, 128, 6, 64]);  view_809 = None
    permute_597: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_811: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_597, [6, 128, 64]);  permute_597 = None
    bmm_88: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_598, view_811);  permute_598 = None
    bmm_89: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_811, permute_599);  view_811 = permute_599 = None
    view_812: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_88, [1, 6, 128, 64]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_242: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_812);  tangents_14 = view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_813: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_89, [1, 6, 128, 128]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_42: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
    mul_463: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_464: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_813, mul_463);  view_813 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_465: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_464, alias_117);  mul_464 = None
    sum_73: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [-1], True)
    mul_466: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_117, sum_73);  alias_117 = sum_73 = None
    sub_48: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_11: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_48, 0);  sub_48 = None
    as_strided_scatter_20: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_11, [6, 128, 128], [16384, 128, 1], 0);  squeeze_11 = None
    as_strided_73: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_10: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_73, [6, 128, 128], [16384, 128, 1])
    as_strided_75: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_73, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_21: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_73, as_strided_75, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_73 = as_strided_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_90: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_600, as_strided_scatter_21);  permute_600 = None
    bmm_91: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_21, permute_601);  as_strided_scatter_21 = permute_601 = None
    view_814: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_90, [1, 6, 64, 128]);  bmm_90 = None
    view_815: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_91, [1, 6, 128, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_602: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_814, [0, 1, 3, 2]);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_243: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_13, permute_602);  tangents_13 = permute_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_603: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_242, [0, 2, 1, 3]);  add_242 = None
    clone_100: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
    view_816: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_100, [1, 128, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_817: "f32[128, 384]" = torch.ops.aten.reshape.default(view_816, [128, 384]);  view_816 = None
    permute_604: "f32[384, 128]" = torch.ops.aten.permute.default(view_817, [1, 0])
    mm_265: "f32[384, 512]" = torch.ops.aten.mm.default(permute_604, view_233);  permute_604 = None
    permute_605: "f32[512, 384]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    mm_266: "f32[128, 512]" = torch.ops.aten.mm.default(view_817, permute_606);  view_817 = permute_606 = None
    view_818: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_266, [1, 128, 512]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_244: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_227, view_818);  add_227 = view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_607: "f32[384, 512]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_608: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_243, [0, 2, 1, 3]);  add_243 = None
    clone_101: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_608, memory_format = torch.contiguous_format);  permute_608 = None
    view_819: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_101, [1, 128, 384]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_820: "f32[128, 384]" = torch.ops.aten.reshape.default(view_819, [128, 384]);  view_819 = None
    permute_609: "f32[384, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
    mm_267: "f32[384, 512]" = torch.ops.aten.mm.default(permute_609, view_233);  permute_609 = None
    permute_610: "f32[512, 384]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    mm_268: "f32[128, 512]" = torch.ops.aten.mm.default(view_820, permute_611);  view_820 = permute_611 = None
    view_821: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_268, [1, 128, 512]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_245: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_244, view_821);  add_244 = view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_612: "f32[384, 512]" = torch.ops.aten.permute.default(permute_610, [1, 0]);  permute_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_613: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
    clone_102: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
    view_822: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 384]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_823: "f32[128, 384]" = torch.ops.aten.reshape.default(view_822, [128, 384]);  view_822 = None
    permute_614: "f32[384, 128]" = torch.ops.aten.permute.default(view_823, [1, 0])
    mm_269: "f32[384, 512]" = torch.ops.aten.mm.default(permute_614, view_322);  permute_614 = view_322 = None
    permute_615: "f32[512, 384]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    mm_270: "f32[128, 512]" = torch.ops.aten.mm.default(view_823, permute_616);  view_823 = permute_616 = None
    view_824: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_270, [1, 128, 512]);  mm_270 = None
    permute_617: "f32[384, 512]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_467: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_824, primals_25);  primals_25 = None
    mul_468: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_824, mul_105);  view_824 = mul_105 = None
    sum_74: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 1], True);  mul_468 = None
    view_825: "f32[512]" = torch.ops.aten.reshape.default(sum_74, [512]);  sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_469: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_467, add_87)
    mul_470: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_467, rsqrt_24);  mul_467 = None
    sum_75: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_246: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_241, mul_470);  add_241 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_99: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
    mul_471: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_75, -0.5);  sum_75 = None
    mul_472: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_471, pow_99);  mul_471 = pow_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_113: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_472, [1, 128, 512]);  mul_472 = None
    div_47: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_113, 512);  expand_113 = None
    pow_100: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_87, 1.0);  add_87 = None
    mul_473: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_100, 2.0);  pow_100 = None
    mul_474: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_473);  div_47 = mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_247: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_246, mul_474);  add_246 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_43: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_475: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_476: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_247, mul_475);  mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_826: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_476, [128, 512]);  mul_476 = None
    permute_618: "f32[512, 128]" = torch.ops.aten.permute.default(view_826, [1, 0])
    mm_271: "f32[512, 384]" = torch.ops.aten.mm.default(permute_618, view_320);  permute_618 = view_320 = None
    permute_619: "f32[384, 512]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    mm_272: "f32[128, 384]" = torch.ops.aten.mm.default(view_826, permute_620);  view_826 = permute_620 = None
    view_827: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_272, [1, 128, 384]);  mm_272 = None
    permute_621: "f32[512, 384]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_828: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_827, [1, 128, 6, 64]);  view_827 = None
    permute_622: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_829: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_622, [6, 128, 64]);  permute_622 = None
    bmm_92: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_623, view_829);  permute_623 = None
    bmm_93: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_829, permute_624);  view_829 = permute_624 = None
    view_830: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_92, [1, 6, 128, 64]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_248: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_830);  tangents_12 = view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_831: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_93, [1, 6, 128, 128]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_44: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_477: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_478: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_831, mul_477);  view_831 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_479: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_478, alias_119);  mul_478 = None
    sum_76: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [-1], True)
    mul_480: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_119, sum_76);  alias_119 = sum_76 = None
    sub_49: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_12: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_49, 0);  sub_49 = None
    as_strided_scatter_22: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_12, [6, 128, 128], [16384, 128, 1], 0);  squeeze_12 = None
    as_strided_80: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_22, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_22 = None
    new_empty_strided_11: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_80, [6, 128, 128], [16384, 128, 1])
    as_strided_82: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_80, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_23: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_80, as_strided_82, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_249: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_231, as_strided_82);  add_231 = as_strided_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_94: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_625, as_strided_scatter_23);  permute_625 = None
    bmm_95: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_23, permute_626);  as_strided_scatter_23 = permute_626 = None
    view_832: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_94, [1, 6, 64, 128]);  bmm_94 = None
    view_833: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_95, [1, 6, 128, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_627: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_832, [0, 1, 3, 2]);  view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_250: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_11, permute_627);  tangents_11 = permute_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_628: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_248, [0, 2, 1, 3]);  add_248 = None
    clone_106: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
    view_834: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 384]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_835: "f32[128, 384]" = torch.ops.aten.reshape.default(view_834, [128, 384]);  view_834 = None
    permute_629: "f32[384, 128]" = torch.ops.aten.permute.default(view_835, [1, 0])
    mm_273: "f32[384, 512]" = torch.ops.aten.mm.default(permute_629, view_302);  permute_629 = None
    permute_630: "f32[512, 384]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    mm_274: "f32[128, 512]" = torch.ops.aten.mm.default(view_835, permute_631);  view_835 = permute_631 = None
    view_836: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_274, [1, 128, 512]);  mm_274 = None
    permute_632: "f32[384, 512]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_633: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_250, [0, 2, 1, 3]);  add_250 = None
    clone_107: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_633, memory_format = torch.contiguous_format);  permute_633 = None
    view_837: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_107, [1, 128, 384]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_838: "f32[128, 384]" = torch.ops.aten.reshape.default(view_837, [128, 384]);  view_837 = None
    permute_634: "f32[384, 128]" = torch.ops.aten.permute.default(view_838, [1, 0])
    mm_275: "f32[384, 512]" = torch.ops.aten.mm.default(permute_634, view_302);  permute_634 = None
    permute_635: "f32[512, 384]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    mm_276: "f32[128, 512]" = torch.ops.aten.mm.default(view_838, permute_636);  view_838 = permute_636 = None
    view_839: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_276, [1, 128, 512]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_251: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_836, view_839);  view_836 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_637: "f32[384, 512]" = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_638: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
    clone_108: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_638, memory_format = torch.contiguous_format);  permute_638 = None
    view_840: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_108, [1, 128, 384]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_841: "f32[128, 384]" = torch.ops.aten.reshape.default(view_840, [128, 384]);  view_840 = None
    permute_639: "f32[384, 128]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_277: "f32[384, 512]" = torch.ops.aten.mm.default(permute_639, view_302);  permute_639 = view_302 = None
    permute_640: "f32[512, 384]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    mm_278: "f32[128, 512]" = torch.ops.aten.mm.default(view_841, permute_641);  view_841 = permute_641 = None
    view_842: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_278, [1, 128, 512]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_252: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_251, view_842);  add_251 = view_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_642: "f32[384, 512]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_481: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_252, primals_24);  primals_24 = None
    mul_482: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_252, mul_103);  add_252 = mul_103 = None
    sum_77: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1], True);  mul_482 = None
    view_843: "f32[512]" = torch.ops.aten.reshape.default(sum_77, [512]);  sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_483: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_481, add_84)
    mul_484: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_481, rsqrt_23);  mul_481 = None
    sum_78: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_253: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_247, mul_484);  add_247 = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_101: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
    mul_485: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_78, -0.5);  sum_78 = None
    mul_486: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_485, pow_101);  mul_485 = pow_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_114: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_486, [1, 128, 512]);  mul_486 = None
    div_48: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_114, 512);  expand_114 = None
    pow_102: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
    mul_487: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_102, 2.0);  pow_102 = None
    mul_488: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_487);  div_48 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_254: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_253, mul_488);  add_253 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_45: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
    mul_489: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_490: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_254, mul_489);  mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_844: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_490, [128, 512]);  mul_490 = None
    permute_643: "f32[512, 128]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_279: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_643, view_300);  permute_643 = view_300 = None
    permute_644: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    mm_280: "f32[128, 1024]" = torch.ops.aten.mm.default(view_844, permute_645);  view_844 = permute_645 = None
    view_845: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_280, [1, 128, 1024]);  mm_280 = None
    permute_646: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_491: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_492: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_845, mul_491);  view_845 = mul_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_493: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_492, mul_101);  mul_101 = None
    mul_494: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_492, view_299);  mul_492 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_846: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_493, [128, 1024]);  mul_493 = None
    permute_647: "f32[1024, 128]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_281: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_647, view_296);  permute_647 = None
    permute_648: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    mm_282: "f32[128, 512]" = torch.ops.aten.mm.default(view_846, permute_649);  view_846 = permute_649 = None
    view_847: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_282, [1, 128, 512]);  mm_282 = None
    permute_650: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_495: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_494, mul_98);  mul_98 = None
    mul_496: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_494, add_83);  mul_494 = add_83 = None
    mul_497: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
    sub_50: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_497);  mul_497 = None
    mul_498: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_495, sub_50);  mul_495 = sub_50 = None
    mul_499: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_498, 0.7978845608028654);  mul_498 = None
    mul_500: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_499, 0.044715)
    pow_103: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_297, 2.0);  view_297 = None
    mul_501: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_103, 3.0);  pow_103 = None
    mul_502: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_255: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_499, mul_502);  mul_499 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_503: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_496, 0.5);  mul_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_256: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_255, mul_503);  add_255 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_848: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_256, [128, 1024]);  add_256 = None
    permute_651: "f32[1024, 128]" = torch.ops.aten.permute.default(view_848, [1, 0])
    mm_283: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_651, view_296);  permute_651 = view_296 = None
    permute_652: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    mm_284: "f32[128, 512]" = torch.ops.aten.mm.default(view_848, permute_653);  view_848 = permute_653 = None
    view_849: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_284, [1, 128, 512]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_257: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_847, view_849);  view_847 = view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_654: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_504: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_257, primals_23);  primals_23 = None
    mul_505: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_257, mul_96);  add_257 = mul_96 = None
    sum_79: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 1], True);  mul_505 = None
    view_850: "f32[512]" = torch.ops.aten.reshape.default(sum_79, [512]);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_506: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_504, add_80)
    mul_507: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_504, rsqrt_22);  mul_504 = None
    sum_80: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True);  mul_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_258: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_254, mul_507);  add_254 = mul_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_104: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
    mul_508: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
    mul_509: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_508, pow_104);  mul_508 = pow_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_115: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_509, [1, 128, 512]);  mul_509 = None
    div_49: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_115, 512);  expand_115 = None
    pow_105: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_80, 1.0);  add_80 = None
    mul_510: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_105, 2.0);  pow_105 = None
    mul_511: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_510);  div_49 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_259: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_258, mul_511);  add_258 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_47: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
    mul_512: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_513: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_259, mul_512);  mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_851: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_513, [128, 512]);  mul_513 = None
    permute_655: "f32[512, 128]" = torch.ops.aten.permute.default(view_851, [1, 0])
    mm_285: "f32[512, 384]" = torch.ops.aten.mm.default(permute_655, view_294);  permute_655 = view_294 = None
    permute_656: "f32[384, 512]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    mm_286: "f32[128, 384]" = torch.ops.aten.mm.default(view_851, permute_657);  view_851 = permute_657 = None
    view_852: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_286, [1, 128, 384]);  mm_286 = None
    permute_658: "f32[512, 384]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_853: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_852, [1, 128, 6, 64]);  view_852 = None
    permute_659: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_854: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_659, [6, 128, 64]);  permute_659 = None
    bmm_96: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_660, view_854);  permute_660 = None
    bmm_97: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_854, permute_661);  view_854 = permute_661 = None
    view_855: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_96, [1, 6, 128, 64]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_260: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_855);  tangents_10 = view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_856: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_97, [1, 6, 128, 128]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_48: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_514: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_515: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_856, mul_514);  view_856 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_516: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_515, alias_123);  mul_515 = None
    sum_81: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [-1], True)
    mul_517: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_123, sum_81);  alias_123 = sum_81 = None
    sub_51: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_13: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_51, 0);  sub_51 = None
    as_strided_scatter_24: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_13, [6, 128, 128], [16384, 128, 1], 0);  squeeze_13 = None
    as_strided_87: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_24, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_24 = None
    new_empty_strided_12: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_87, [6, 128, 128], [16384, 128, 1])
    as_strided_89: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_87, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_25: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_87, as_strided_89, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_87 = as_strided_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_98: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_662, as_strided_scatter_25);  permute_662 = None
    bmm_99: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_25, permute_663);  as_strided_scatter_25 = permute_663 = None
    view_857: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_98, [1, 6, 64, 128]);  bmm_98 = None
    view_858: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_99, [1, 6, 128, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_664: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_857, [0, 1, 3, 2]);  view_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_261: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_9, permute_664);  tangents_9 = permute_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_665: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_260, [0, 2, 1, 3]);  add_260 = None
    clone_114: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
    view_859: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 384]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_860: "f32[128, 384]" = torch.ops.aten.reshape.default(view_859, [128, 384]);  view_859 = None
    permute_666: "f32[384, 128]" = torch.ops.aten.permute.default(view_860, [1, 0])
    mm_287: "f32[384, 512]" = torch.ops.aten.mm.default(permute_666, view_233);  permute_666 = None
    permute_667: "f32[512, 384]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    mm_288: "f32[128, 512]" = torch.ops.aten.mm.default(view_860, permute_668);  view_860 = permute_668 = None
    view_861: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_288, [1, 128, 512]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_262: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_245, view_861);  add_245 = view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_669: "f32[384, 512]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_670: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_261, [0, 2, 1, 3]);  add_261 = None
    clone_115: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
    view_862: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_115, [1, 128, 384]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_863: "f32[128, 384]" = torch.ops.aten.reshape.default(view_862, [128, 384]);  view_862 = None
    permute_671: "f32[384, 128]" = torch.ops.aten.permute.default(view_863, [1, 0])
    mm_289: "f32[384, 512]" = torch.ops.aten.mm.default(permute_671, view_233);  permute_671 = None
    permute_672: "f32[512, 384]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    mm_290: "f32[128, 512]" = torch.ops.aten.mm.default(view_863, permute_673);  view_863 = permute_673 = None
    view_864: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_290, [1, 128, 512]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_263: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_262, view_864);  add_262 = view_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_674: "f32[384, 512]" = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_675: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
    clone_116: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_675, memory_format = torch.contiguous_format);  permute_675 = None
    view_865: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_116, [1, 128, 384]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_866: "f32[128, 384]" = torch.ops.aten.reshape.default(view_865, [128, 384]);  view_865 = None
    permute_676: "f32[384, 128]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_291: "f32[384, 512]" = torch.ops.aten.mm.default(permute_676, view_276);  permute_676 = view_276 = None
    permute_677: "f32[512, 384]" = torch.ops.aten.permute.default(mm_291, [1, 0]);  mm_291 = None
    mm_292: "f32[128, 512]" = torch.ops.aten.mm.default(view_866, permute_678);  view_866 = permute_678 = None
    view_867: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_292, [1, 128, 512]);  mm_292 = None
    permute_679: "f32[384, 512]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_518: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_867, primals_22);  primals_22 = None
    mul_519: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_867, mul_94);  view_867 = mul_94 = None
    sum_82: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 1], True);  mul_519 = None
    view_868: "f32[512]" = torch.ops.aten.reshape.default(sum_82, [512]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_520: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_518, add_77)
    mul_521: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_518, rsqrt_21);  mul_518 = None
    sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True);  mul_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_264: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_259, mul_521);  add_259 = mul_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_106: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
    mul_522: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_83, -0.5);  sum_83 = None
    mul_523: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_522, pow_106);  mul_522 = pow_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_116: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_523, [1, 128, 512]);  mul_523 = None
    div_50: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_116, 512);  expand_116 = None
    pow_107: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_77, 1.0);  add_77 = None
    mul_524: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_107, 2.0);  pow_107 = None
    mul_525: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_524);  div_50 = mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_265: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_264, mul_525);  add_264 = mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_49: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_526: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_527: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_265, mul_526);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_869: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_527, [128, 512]);  mul_527 = None
    permute_680: "f32[512, 128]" = torch.ops.aten.permute.default(view_869, [1, 0])
    mm_293: "f32[512, 384]" = torch.ops.aten.mm.default(permute_680, view_274);  permute_680 = view_274 = None
    permute_681: "f32[384, 512]" = torch.ops.aten.permute.default(mm_293, [1, 0]);  mm_293 = None
    mm_294: "f32[128, 384]" = torch.ops.aten.mm.default(view_869, permute_682);  view_869 = permute_682 = None
    view_870: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_294, [1, 128, 384]);  mm_294 = None
    permute_683: "f32[512, 384]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_871: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_870, [1, 128, 6, 64]);  view_870 = None
    permute_684: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_872: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_684, [6, 128, 64]);  permute_684 = None
    bmm_100: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_685, view_872);  permute_685 = None
    bmm_101: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_872, permute_686);  view_872 = permute_686 = None
    view_873: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_100, [1, 6, 128, 64]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_266: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_873);  tangents_8 = view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_874: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_101, [1, 6, 128, 128]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_50: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
    mul_528: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_529: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_874, mul_528);  view_874 = mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_530: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_529, alias_125);  mul_529 = None
    sum_84: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [-1], True)
    mul_531: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_125, sum_84);  alias_125 = sum_84 = None
    sub_52: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_14: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_52, 0);  sub_52 = None
    as_strided_scatter_26: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_14, [6, 128, 128], [16384, 128, 1], 0);  squeeze_14 = None
    as_strided_94: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_26, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_26 = None
    new_empty_strided_13: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_94, [6, 128, 128], [16384, 128, 1])
    as_strided_96: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_94, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_27: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_94, as_strided_96, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_267: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_249, as_strided_96);  add_249 = as_strided_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_102: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_687, as_strided_scatter_27);  permute_687 = None
    bmm_103: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_27, permute_688);  as_strided_scatter_27 = permute_688 = None
    view_875: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_102, [1, 6, 64, 128]);  bmm_102 = None
    view_876: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_103, [1, 6, 128, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_689: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_875, [0, 1, 3, 2]);  view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_268: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_7, permute_689);  tangents_7 = permute_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_690: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_266, [0, 2, 1, 3]);  add_266 = None
    clone_120: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
    view_877: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_120, [1, 128, 384]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_878: "f32[128, 384]" = torch.ops.aten.reshape.default(view_877, [128, 384]);  view_877 = None
    permute_691: "f32[384, 128]" = torch.ops.aten.permute.default(view_878, [1, 0])
    mm_295: "f32[384, 512]" = torch.ops.aten.mm.default(permute_691, view_256);  permute_691 = None
    permute_692: "f32[512, 384]" = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
    mm_296: "f32[128, 512]" = torch.ops.aten.mm.default(view_878, permute_693);  view_878 = permute_693 = None
    view_879: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_296, [1, 128, 512]);  mm_296 = None
    permute_694: "f32[384, 512]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_695: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_268, [0, 2, 1, 3]);  add_268 = None
    clone_121: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
    view_880: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 384]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_881: "f32[128, 384]" = torch.ops.aten.reshape.default(view_880, [128, 384]);  view_880 = None
    permute_696: "f32[384, 128]" = torch.ops.aten.permute.default(view_881, [1, 0])
    mm_297: "f32[384, 512]" = torch.ops.aten.mm.default(permute_696, view_256);  permute_696 = None
    permute_697: "f32[512, 384]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    mm_298: "f32[128, 512]" = torch.ops.aten.mm.default(view_881, permute_698);  view_881 = permute_698 = None
    view_882: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_298, [1, 128, 512]);  mm_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_269: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_879, view_882);  view_879 = view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_699: "f32[384, 512]" = torch.ops.aten.permute.default(permute_697, [1, 0]);  permute_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_700: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_876, [0, 2, 1, 3]);  view_876 = None
    clone_122: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_700, memory_format = torch.contiguous_format);  permute_700 = None
    view_883: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 384]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_884: "f32[128, 384]" = torch.ops.aten.reshape.default(view_883, [128, 384]);  view_883 = None
    permute_701: "f32[384, 128]" = torch.ops.aten.permute.default(view_884, [1, 0])
    mm_299: "f32[384, 512]" = torch.ops.aten.mm.default(permute_701, view_256);  permute_701 = view_256 = None
    permute_702: "f32[512, 384]" = torch.ops.aten.permute.default(mm_299, [1, 0]);  mm_299 = None
    mm_300: "f32[128, 512]" = torch.ops.aten.mm.default(view_884, permute_703);  view_884 = permute_703 = None
    view_885: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_300, [1, 128, 512]);  mm_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_270: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_269, view_885);  add_269 = view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_704: "f32[384, 512]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_532: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_270, primals_21);  primals_21 = None
    mul_533: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_270, mul_92);  add_270 = mul_92 = None
    sum_85: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 1], True);  mul_533 = None
    view_886: "f32[512]" = torch.ops.aten.reshape.default(sum_85, [512]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_534: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_532, add_74)
    mul_535: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_532, rsqrt_20);  mul_532 = None
    sum_86: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True);  mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_271: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_265, mul_535);  add_265 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_108: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
    mul_536: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
    mul_537: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_536, pow_108);  mul_536 = pow_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_117: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_537, [1, 128, 512]);  mul_537 = None
    div_51: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_117, 512);  expand_117 = None
    pow_109: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 1.0);  add_74 = None
    mul_538: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_109, 2.0);  pow_109 = None
    mul_539: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_538);  div_51 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_272: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_271, mul_539);  add_271 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_51: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_540: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_541: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_272, mul_540);  mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_887: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_541, [128, 512]);  mul_541 = None
    permute_705: "f32[512, 128]" = torch.ops.aten.permute.default(view_887, [1, 0])
    mm_301: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_705, view_254);  permute_705 = view_254 = None
    permute_706: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_301, [1, 0]);  mm_301 = None
    mm_302: "f32[128, 1024]" = torch.ops.aten.mm.default(view_887, permute_707);  view_887 = permute_707 = None
    view_888: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_302, [1, 128, 1024]);  mm_302 = None
    permute_708: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_52: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_542: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_543: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_888, mul_542);  view_888 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_544: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_543, mul_90);  mul_90 = None
    mul_545: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_543, view_253);  mul_543 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_889: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_544, [128, 1024]);  mul_544 = None
    permute_709: "f32[1024, 128]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_303: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_709, view_250);  permute_709 = None
    permute_710: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_303, [1, 0]);  mm_303 = None
    mm_304: "f32[128, 512]" = torch.ops.aten.mm.default(view_889, permute_711);  view_889 = permute_711 = None
    view_890: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_304, [1, 128, 512]);  mm_304 = None
    permute_712: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_710, [1, 0]);  permute_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_546: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_545, mul_87);  mul_87 = None
    mul_547: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_545, add_73);  mul_545 = add_73 = None
    mul_548: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
    sub_53: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_548);  mul_548 = None
    mul_549: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_546, sub_53);  mul_546 = sub_53 = None
    mul_550: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_549, 0.7978845608028654);  mul_549 = None
    mul_551: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_550, 0.044715)
    pow_110: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 2.0);  view_251 = None
    mul_552: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_110, 3.0);  pow_110 = None
    mul_553: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_551, mul_552);  mul_551 = mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_273: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_550, mul_553);  mul_550 = mul_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_554: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_547, 0.5);  mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_274: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_273, mul_554);  add_273 = mul_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_891: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_274, [128, 1024]);  add_274 = None
    permute_713: "f32[1024, 128]" = torch.ops.aten.permute.default(view_891, [1, 0])
    mm_305: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_713, view_250);  permute_713 = view_250 = None
    permute_714: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_305, [1, 0]);  mm_305 = None
    mm_306: "f32[128, 512]" = torch.ops.aten.mm.default(view_891, permute_715);  view_891 = permute_715 = None
    view_892: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_306, [1, 128, 512]);  mm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_275: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_890, view_892);  view_890 = view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_716: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_555: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_275, primals_20);  primals_20 = None
    mul_556: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_275, mul_85);  add_275 = mul_85 = None
    sum_87: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1], True);  mul_556 = None
    view_893: "f32[512]" = torch.ops.aten.reshape.default(sum_87, [512]);  sum_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_557: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_555, add_70)
    mul_558: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_555, rsqrt_19);  mul_555 = None
    sum_88: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_276: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_272, mul_558);  add_272 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_111: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
    mul_559: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_88, -0.5);  sum_88 = None
    mul_560: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_559, pow_111);  mul_559 = pow_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_118: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_560, [1, 128, 512]);  mul_560 = None
    div_52: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_118, 512);  expand_118 = None
    pow_112: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
    mul_561: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_112, 2.0);  pow_112 = None
    mul_562: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_561);  div_52 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_277: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_276, mul_562);  add_276 = mul_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    convert_element_type_53: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_563: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_564: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_277, mul_563);  mul_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_894: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_564, [128, 512]);  mul_564 = None
    permute_717: "f32[512, 128]" = torch.ops.aten.permute.default(view_894, [1, 0])
    mm_307: "f32[512, 384]" = torch.ops.aten.mm.default(permute_717, view_248);  permute_717 = view_248 = None
    permute_718: "f32[384, 512]" = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
    mm_308: "f32[128, 384]" = torch.ops.aten.mm.default(view_894, permute_719);  view_894 = permute_719 = None
    view_895: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_308, [1, 128, 384]);  mm_308 = None
    permute_720: "f32[512, 384]" = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_896: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_895, [1, 128, 6, 64]);  view_895 = None
    permute_721: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_897: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_721, [6, 128, 64]);  permute_721 = None
    bmm_104: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_722, view_897);  permute_722 = None
    bmm_105: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_897, permute_723);  view_897 = permute_723 = None
    view_898: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_104, [1, 6, 128, 64]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_278: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_898);  tangents_6 = view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_899: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_105, [1, 6, 128, 128]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_54: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_565: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_566: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_899, mul_565);  view_899 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_567: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_566, alias_129);  mul_566 = None
    sum_89: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [-1], True)
    mul_568: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_129, sum_89);  alias_129 = sum_89 = None
    sub_54: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_15: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_54, 0);  sub_54 = None
    as_strided_scatter_28: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_15, [6, 128, 128], [16384, 128, 1], 0);  squeeze_15 = None
    as_strided_101: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_14: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_101, [6, 128, 128], [16384, 128, 1])
    as_strided_103: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_101, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_29: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_101, as_strided_103, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_101 = as_strided_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_106: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_724, as_strided_scatter_29);  permute_724 = None
    bmm_107: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_29, permute_725);  as_strided_scatter_29 = permute_725 = None
    view_900: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_106, [1, 6, 64, 128]);  bmm_106 = None
    view_901: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_107, [1, 6, 128, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_726: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2]);  view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_279: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_5, permute_726);  tangents_5 = permute_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_727: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_278, [0, 2, 1, 3]);  add_278 = None
    clone_128: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_727, memory_format = torch.contiguous_format);  permute_727 = None
    view_902: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_128, [1, 128, 384]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_903: "f32[128, 384]" = torch.ops.aten.reshape.default(view_902, [128, 384]);  view_902 = None
    permute_728: "f32[384, 128]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_309: "f32[384, 512]" = torch.ops.aten.mm.default(permute_728, view_233);  permute_728 = None
    permute_729: "f32[512, 384]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    mm_310: "f32[128, 512]" = torch.ops.aten.mm.default(view_903, permute_730);  view_903 = permute_730 = None
    view_904: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_310, [1, 128, 512]);  mm_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_280: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_263, view_904);  add_263 = view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_731: "f32[384, 512]" = torch.ops.aten.permute.default(permute_729, [1, 0]);  permute_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_732: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_279, [0, 2, 1, 3]);  add_279 = None
    clone_129: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_732, memory_format = torch.contiguous_format);  permute_732 = None
    view_905: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 384]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_906: "f32[128, 384]" = torch.ops.aten.reshape.default(view_905, [128, 384]);  view_905 = None
    permute_733: "f32[384, 128]" = torch.ops.aten.permute.default(view_906, [1, 0])
    mm_311: "f32[384, 512]" = torch.ops.aten.mm.default(permute_733, view_233);  permute_733 = view_233 = None
    permute_734: "f32[512, 384]" = torch.ops.aten.permute.default(mm_311, [1, 0]);  mm_311 = None
    mm_312: "f32[128, 512]" = torch.ops.aten.mm.default(view_906, permute_735);  view_906 = permute_735 = None
    view_907: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_312, [1, 128, 512]);  mm_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    add_281: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_280, view_907);  add_280 = view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_736: "f32[384, 512]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_737: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_901, [0, 2, 1, 3]);  view_901 = None
    clone_130: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_737, memory_format = torch.contiguous_format);  permute_737 = None
    view_908: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 384]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_909: "f32[128, 384]" = torch.ops.aten.reshape.default(view_908, [128, 384]);  view_908 = None
    permute_738: "f32[384, 128]" = torch.ops.aten.permute.default(view_909, [1, 0])
    mm_313: "f32[384, 512]" = torch.ops.aten.mm.default(permute_738, view_230);  permute_738 = view_230 = None
    permute_739: "f32[512, 384]" = torch.ops.aten.permute.default(mm_313, [1, 0]);  mm_313 = None
    mm_314: "f32[128, 512]" = torch.ops.aten.mm.default(view_909, permute_740);  view_909 = permute_740 = None
    view_910: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_314, [1, 128, 512]);  mm_314 = None
    permute_741: "f32[384, 512]" = torch.ops.aten.permute.default(permute_739, [1, 0]);  permute_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_569: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_910, primals_19);  primals_19 = None
    mul_570: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_910, mul_83);  view_910 = mul_83 = None
    sum_90: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1], True);  mul_570 = None
    view_911: "f32[512]" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_571: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_569, add_66)
    mul_572: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_569, rsqrt_18);  mul_569 = None
    sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_282: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_277, mul_572);  add_277 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_113: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
    mul_573: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_91, -0.5);  sum_91 = None
    mul_574: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_573, pow_113);  mul_573 = pow_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_119: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_574, [1, 128, 512]);  mul_574 = None
    div_53: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_119, 512);  expand_119 = None
    pow_114: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 1.0);  add_66 = None
    mul_575: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_114, 2.0);  pow_114 = None
    mul_576: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_575);  div_53 = mul_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_283: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_282, mul_576);  add_282 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_55: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_577: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_578: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_283, mul_577);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_912: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_578, [128, 512]);  mul_578 = None
    permute_742: "f32[512, 128]" = torch.ops.aten.permute.default(view_912, [1, 0])
    mm_315: "f32[512, 384]" = torch.ops.aten.mm.default(permute_742, view_228);  permute_742 = view_228 = None
    permute_743: "f32[384, 512]" = torch.ops.aten.permute.default(mm_315, [1, 0]);  mm_315 = None
    mm_316: "f32[128, 384]" = torch.ops.aten.mm.default(view_912, permute_744);  view_912 = permute_744 = None
    view_913: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_316, [1, 128, 384]);  mm_316 = None
    permute_745: "f32[512, 384]" = torch.ops.aten.permute.default(permute_743, [1, 0]);  permute_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_914: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_913, [1, 128, 6, 64]);  view_913 = None
    permute_746: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_915: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_746, [6, 128, 64]);  permute_746 = None
    bmm_108: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_747, view_915);  permute_747 = None
    bmm_109: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_915, permute_748);  view_915 = permute_748 = None
    view_916: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_108, [1, 6, 128, 64]);  bmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    add_284: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_916);  tangents_4 = view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_917: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, [1, 6, 128, 128]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_56: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_579: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_580: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_917, mul_579);  view_917 = mul_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_581: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_580, alias_131);  mul_580 = None
    sum_92: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [-1], True)
    mul_582: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_131, sum_92);  alias_131 = sum_92 = None
    sub_55: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_16: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_55, 0);  sub_55 = None
    as_strided_scatter_30: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_16, [6, 128, 128], [16384, 128, 1], 0);  squeeze_16 = None
    as_strided_108: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_15: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_108, [6, 128, 128], [16384, 128, 1])
    as_strided_110: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_108, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_31: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_108, as_strided_110, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_285: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_267, as_strided_110);  add_267 = as_strided_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_17: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(add_285, 0);  add_285 = None
    permute_749: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    full_default_11: "b8[128, 128, 1]" = torch.ops.aten.full.default([128, 128, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_6: "f32[128, 128, 6]" = torch.ops.aten.where.self(full_default_11, full_default_7, permute_749);  permute_749 = None
    clone_134: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_6, memory_format = torch.contiguous_format);  where_6 = None
    full_default_13: "f32[32, 6]" = torch.ops.aten.full.default([32, 6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[32, 6]" = torch.ops.aten._unsafe_index_put.default(full_default_13, [add_63], clone_134, True);  add_63 = clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_110: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_750, as_strided_scatter_31);  permute_750 = None
    bmm_111: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_31, permute_751);  as_strided_scatter_31 = permute_751 = None
    view_918: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_110, [1, 6, 64, 128]);  bmm_110 = None
    view_919: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_111, [1, 6, 128, 64]);  bmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_752: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_918, [0, 1, 3, 2]);  view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    add_286: "f32[1, 6, 128, 64]" = torch.ops.aten.add.Tensor(tangents_3, permute_752);  tangents_3 = permute_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_753: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_284, [0, 2, 1, 3]);  add_284 = None
    clone_135: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_753, memory_format = torch.contiguous_format);  permute_753 = None
    view_920: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_135, [1, 128, 384]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_921: "f32[128, 384]" = torch.ops.aten.reshape.default(view_920, [128, 384]);  view_920 = None
    permute_754: "f32[384, 128]" = torch.ops.aten.permute.default(view_921, [1, 0])
    mm_317: "f32[384, 512]" = torch.ops.aten.mm.default(permute_754, view_210);  permute_754 = None
    permute_755: "f32[512, 384]" = torch.ops.aten.permute.default(mm_317, [1, 0]);  mm_317 = None
    mm_318: "f32[128, 512]" = torch.ops.aten.mm.default(view_921, permute_756);  view_921 = permute_756 = None
    view_922: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_318, [1, 128, 512]);  mm_318 = None
    permute_757: "f32[384, 512]" = torch.ops.aten.permute.default(permute_755, [1, 0]);  permute_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_758: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(add_286, [0, 2, 1, 3]);  add_286 = None
    clone_136: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_758, memory_format = torch.contiguous_format);  permute_758 = None
    view_923: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_136, [1, 128, 384]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_924: "f32[128, 384]" = torch.ops.aten.reshape.default(view_923, [128, 384]);  view_923 = None
    permute_759: "f32[384, 128]" = torch.ops.aten.permute.default(view_924, [1, 0])
    mm_319: "f32[384, 512]" = torch.ops.aten.mm.default(permute_759, view_210);  permute_759 = None
    permute_760: "f32[512, 384]" = torch.ops.aten.permute.default(mm_319, [1, 0]);  mm_319 = None
    mm_320: "f32[128, 512]" = torch.ops.aten.mm.default(view_924, permute_761);  view_924 = permute_761 = None
    view_925: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_320, [1, 128, 512]);  mm_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_287: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_922, view_925);  view_922 = view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_762: "f32[384, 512]" = torch.ops.aten.permute.default(permute_760, [1, 0]);  permute_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_763: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_919, [0, 2, 1, 3]);  view_919 = None
    clone_137: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_763, memory_format = torch.contiguous_format);  permute_763 = None
    view_926: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 384]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_927: "f32[128, 384]" = torch.ops.aten.reshape.default(view_926, [128, 384]);  view_926 = None
    permute_764: "f32[384, 128]" = torch.ops.aten.permute.default(view_927, [1, 0])
    mm_321: "f32[384, 512]" = torch.ops.aten.mm.default(permute_764, view_210);  permute_764 = view_210 = None
    permute_765: "f32[512, 384]" = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
    mm_322: "f32[128, 512]" = torch.ops.aten.mm.default(view_927, permute_766);  view_927 = permute_766 = None
    view_928: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_322, [1, 128, 512]);  mm_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_288: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_287, view_928);  add_287 = view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_767: "f32[384, 512]" = torch.ops.aten.permute.default(permute_765, [1, 0]);  permute_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_583: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_288, primals_18);  primals_18 = None
    mul_584: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_288, mul_80);  add_288 = mul_80 = None
    sum_93: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1], True);  mul_584 = None
    view_929: "f32[512]" = torch.ops.aten.reshape.default(sum_93, [512]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_585: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_583, getitem_68)
    mul_586: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_583, rsqrt_17);  mul_583 = None
    sum_94: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_585, [2], True);  mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_289: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_283, mul_586);  add_283 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_115: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
    mul_587: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_94, -0.5);  sum_94 = None
    mul_588: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_587, pow_115);  mul_587 = pow_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_120: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_588, [1, 128, 512]);  mul_588 = None
    div_54: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_120, 512);  expand_120 = None
    pow_116: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem_68, 1.0);  getitem_68 = None
    mul_589: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_116, 2.0);  pow_116 = None
    mul_590: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_589);  div_54 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_290: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_289, mul_590);  add_289 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_57: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_69, torch.float32);  getitem_69 = None
    mul_591: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_592: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_290, mul_591);  add_290 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view_209, -1)
    unsqueeze_20: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_7: "f32[1, 128, 512]" = torch.ops.aten.where.self(unsqueeze_20, full_default_7, mul_592);  unsqueeze_20 = mul_592 = None
    full_default_15: "f32[250112, 512]" = torch.ops.aten.full.default([250112, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[250112, 512]" = torch.ops.aten._unsafe_index_put.default(full_default_15, [view_209], where_7, True);  view_209 = where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_58: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_593: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_594: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_281, mul_593);  add_281 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_595: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_594, primals_17);  primals_17 = None
    mul_596: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_594, mul_75);  mul_594 = mul_75 = None
    sum_95: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1], True);  mul_596 = None
    view_930: "f32[512]" = torch.ops.aten.reshape.default(sum_95, [512]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_597: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_595, add_59)
    mul_598: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_595, rsqrt_16);  mul_595 = None
    sum_96: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True);  mul_597 = None
    pow_117: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
    mul_599: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_96, -0.5);  sum_96 = None
    mul_600: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_599, pow_117);  mul_599 = pow_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_121: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_600, [1, 128, 512]);  mul_600 = None
    div_55: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_121, 512);  expand_121 = None
    pow_118: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 1.0);  add_59 = None
    mul_601: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_118, 2.0);  pow_118 = None
    mul_602: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_601);  div_55 = mul_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_291: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_598, mul_602);  mul_598 = mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_59: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_603: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_604: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_291, mul_603);  mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_931: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_604, [128, 512]);  mul_604 = None
    permute_768: "f32[512, 128]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_323: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_768, view_207);  permute_768 = view_207 = None
    permute_769: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_323, [1, 0]);  mm_323 = None
    mm_324: "f32[128, 1024]" = torch.ops.aten.mm.default(view_931, permute_770);  view_931 = permute_770 = None
    view_932: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_324, [1, 128, 1024]);  mm_324 = None
    permute_771: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_769, [1, 0]);  permute_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_60: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
    mul_605: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_606: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_932, mul_605);  view_932 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_607: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_606, mul_73);  mul_73 = None
    mul_608: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_606, view_206);  mul_606 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_933: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_607, [128, 1024]);  mul_607 = None
    permute_772: "f32[1024, 128]" = torch.ops.aten.permute.default(view_933, [1, 0])
    mm_325: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_772, view_203);  permute_772 = None
    permute_773: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
    mm_326: "f32[128, 512]" = torch.ops.aten.mm.default(view_933, permute_774);  view_933 = permute_774 = None
    view_934: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_326, [1, 128, 512]);  mm_326 = None
    permute_775: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_609: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_608, mul_70);  mul_70 = None
    mul_610: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_608, add_58);  mul_608 = add_58 = None
    mul_611: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
    sub_56: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_611);  mul_611 = None
    mul_612: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_609, sub_56);  mul_609 = sub_56 = None
    mul_613: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_612, 0.7978845608028654);  mul_612 = None
    mul_614: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_613, 0.044715)
    pow_119: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_204, 2.0);  view_204 = None
    mul_615: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_119, 3.0);  pow_119 = None
    mul_616: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_292: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_613, mul_616);  mul_613 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_617: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_610, 0.5);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_293: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_292, mul_617);  add_292 = mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_935: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_293, [128, 1024]);  add_293 = None
    permute_776: "f32[1024, 128]" = torch.ops.aten.permute.default(view_935, [1, 0])
    mm_327: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_776, view_203);  permute_776 = view_203 = None
    permute_777: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_327, [1, 0]);  mm_327 = None
    mm_328: "f32[128, 512]" = torch.ops.aten.mm.default(view_935, permute_778);  view_935 = permute_778 = None
    view_936: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_328, [1, 128, 512]);  mm_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_294: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_934, view_936);  view_934 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_779: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_618: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_294, primals_16);  primals_16 = None
    mul_619: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_294, mul_68);  add_294 = mul_68 = None
    sum_97: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 1], True);  mul_619 = None
    view_937: "f32[512]" = torch.ops.aten.reshape.default(sum_97, [512]);  sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_620: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_618, add_55)
    mul_621: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_618, rsqrt_15);  mul_618 = None
    sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_295: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_291, mul_621);  add_291 = mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_120: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
    mul_622: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_98, -0.5);  sum_98 = None
    mul_623: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_622, pow_120);  mul_622 = pow_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_122: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_623, [1, 128, 512]);  mul_623 = None
    div_56: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_122, 512);  expand_122 = None
    pow_121: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_55, 1.0);  add_55 = None
    mul_624: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_121, 2.0);  pow_121 = None
    mul_625: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_56, mul_624);  div_56 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_296: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_295, mul_625);  add_295 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_61: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_626: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_627: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_296, mul_626);  mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_938: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_627, [128, 512]);  mul_627 = None
    permute_780: "f32[512, 128]" = torch.ops.aten.permute.default(view_938, [1, 0])
    mm_329: "f32[512, 384]" = torch.ops.aten.mm.default(permute_780, view_201);  permute_780 = view_201 = None
    permute_781: "f32[384, 512]" = torch.ops.aten.permute.default(mm_329, [1, 0]);  mm_329 = None
    mm_330: "f32[128, 384]" = torch.ops.aten.mm.default(view_938, permute_782);  view_938 = permute_782 = None
    view_939: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_330, [1, 128, 384]);  mm_330 = None
    permute_783: "f32[512, 384]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_940: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_939, [1, 128, 6, 64]);  view_939 = None
    permute_784: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_940, [0, 2, 1, 3]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_941: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_784, [6, 128, 64]);  permute_784 = None
    bmm_112: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_785, view_941);  permute_785 = None
    bmm_113: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_941, permute_786);  view_941 = permute_786 = None
    view_942: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_112, [1, 6, 128, 64]);  bmm_112 = None
    view_943: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_113, [1, 6, 128, 128]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_62: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None
    mul_628: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_629: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_943, mul_628);  view_943 = mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_630: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_629, alias_136);  mul_629 = None
    sum_99: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [-1], True)
    mul_631: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_136, sum_99);  alias_136 = sum_99 = None
    sub_57: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_18: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_57, 0);  sub_57 = None
    as_strided_scatter_32: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_18, [6, 128, 128], [16384, 128, 1], 0);  squeeze_18 = None
    as_strided_115: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_32, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_32 = None
    new_empty_strided_16: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_115, [6, 128, 128], [16384, 128, 1])
    as_strided_117: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_115, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_33: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_115, as_strided_117, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_114: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_787, as_strided_scatter_33);  permute_787 = None
    bmm_115: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_33, permute_788);  as_strided_scatter_33 = permute_788 = None
    view_944: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_114, [1, 6, 64, 128]);  bmm_114 = None
    view_945: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_115, [1, 6, 128, 64]);  bmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_789: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_944, [0, 1, 3, 2]);  view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_790: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
    clone_145: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
    view_946: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 384]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_947: "f32[128, 384]" = torch.ops.aten.reshape.default(view_946, [128, 384]);  view_946 = None
    permute_791: "f32[384, 128]" = torch.ops.aten.permute.default(view_947, [1, 0])
    mm_331: "f32[384, 512]" = torch.ops.aten.mm.default(permute_791, view_183);  permute_791 = None
    permute_792: "f32[512, 384]" = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
    mm_332: "f32[128, 512]" = torch.ops.aten.mm.default(view_947, permute_793);  view_947 = permute_793 = None
    view_948: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_332, [1, 128, 512]);  mm_332 = None
    permute_794: "f32[384, 512]" = torch.ops.aten.permute.default(permute_792, [1, 0]);  permute_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_795: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_789, [0, 2, 1, 3]);  permute_789 = None
    view_949: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_795, [1, 128, 384]);  permute_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_950: "f32[128, 384]" = torch.ops.aten.reshape.default(view_949, [128, 384]);  view_949 = None
    permute_796: "f32[384, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_333: "f32[384, 512]" = torch.ops.aten.mm.default(permute_796, view_183);  permute_796 = None
    permute_797: "f32[512, 384]" = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
    mm_334: "f32[128, 512]" = torch.ops.aten.mm.default(view_950, permute_798);  view_950 = permute_798 = None
    view_951: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_334, [1, 128, 512]);  mm_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_297: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_948, view_951);  view_948 = view_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_799: "f32[384, 512]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_800: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_945, [0, 2, 1, 3]);  view_945 = None
    clone_146: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_800, memory_format = torch.contiguous_format);  permute_800 = None
    view_952: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 384]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_953: "f32[128, 384]" = torch.ops.aten.reshape.default(view_952, [128, 384]);  view_952 = None
    permute_801: "f32[384, 128]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_335: "f32[384, 512]" = torch.ops.aten.mm.default(permute_801, view_183);  permute_801 = view_183 = None
    permute_802: "f32[512, 384]" = torch.ops.aten.permute.default(mm_335, [1, 0]);  mm_335 = None
    mm_336: "f32[128, 512]" = torch.ops.aten.mm.default(view_953, permute_803);  view_953 = permute_803 = None
    view_954: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_336, [1, 128, 512]);  mm_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_298: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_297, view_954);  add_297 = view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_804: "f32[384, 512]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_632: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_298, primals_15);  primals_15 = None
    mul_633: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_298, mul_66);  add_298 = mul_66 = None
    sum_100: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 1], True);  mul_633 = None
    view_955: "f32[512]" = torch.ops.aten.reshape.default(sum_100, [512]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_634: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_632, add_52)
    mul_635: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_632, rsqrt_14);  mul_632 = None
    sum_101: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_299: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_296, mul_635);  add_296 = mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_122: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
    mul_636: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_101, -0.5);  sum_101 = None
    mul_637: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_636, pow_122);  mul_636 = pow_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_123: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_637, [1, 128, 512]);  mul_637 = None
    div_57: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_123, 512);  expand_123 = None
    pow_123: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
    mul_638: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_123, 2.0);  pow_123 = None
    mul_639: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_57, mul_638);  div_57 = mul_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_300: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_299, mul_639);  add_299 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_63: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_640: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_641: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_300, mul_640);  mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_956: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_641, [128, 512]);  mul_641 = None
    permute_805: "f32[512, 128]" = torch.ops.aten.permute.default(view_956, [1, 0])
    mm_337: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_805, view_181);  permute_805 = view_181 = None
    permute_806: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_337, [1, 0]);  mm_337 = None
    mm_338: "f32[128, 1024]" = torch.ops.aten.mm.default(view_956, permute_807);  view_956 = permute_807 = None
    view_957: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_338, [1, 128, 1024]);  mm_338 = None
    permute_808: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_806, [1, 0]);  permute_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_642: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_643: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_957, mul_642);  view_957 = mul_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_644: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_643, mul_64);  mul_64 = None
    mul_645: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_643, view_180);  mul_643 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_958: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_644, [128, 1024]);  mul_644 = None
    permute_809: "f32[1024, 128]" = torch.ops.aten.permute.default(view_958, [1, 0])
    mm_339: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_809, view_177);  permute_809 = None
    permute_810: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_339, [1, 0]);  mm_339 = None
    mm_340: "f32[128, 512]" = torch.ops.aten.mm.default(view_958, permute_811);  view_958 = permute_811 = None
    view_959: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_340, [1, 128, 512]);  mm_340 = None
    permute_812: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_646: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_645, mul_61);  mul_61 = None
    mul_647: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_645, add_51);  mul_645 = add_51 = None
    mul_648: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
    sub_58: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_648);  mul_648 = None
    mul_649: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_646, sub_58);  mul_646 = sub_58 = None
    mul_650: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_649, 0.7978845608028654);  mul_649 = None
    mul_651: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_650, 0.044715)
    pow_124: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 2.0);  view_178 = None
    mul_652: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_124, 3.0);  pow_124 = None
    mul_653: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_301: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_650, mul_653);  mul_650 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_654: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_647, 0.5);  mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_302: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_301, mul_654);  add_301 = mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_960: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_302, [128, 1024]);  add_302 = None
    permute_813: "f32[1024, 128]" = torch.ops.aten.permute.default(view_960, [1, 0])
    mm_341: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_813, view_177);  permute_813 = view_177 = None
    permute_814: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_341, [1, 0]);  mm_341 = None
    mm_342: "f32[128, 512]" = torch.ops.aten.mm.default(view_960, permute_815);  view_960 = permute_815 = None
    view_961: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_342, [1, 128, 512]);  mm_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_303: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_959, view_961);  view_959 = view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_816: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_655: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_303, primals_14);  primals_14 = None
    mul_656: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_303, mul_59);  add_303 = mul_59 = None
    sum_102: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 1], True);  mul_656 = None
    view_962: "f32[512]" = torch.ops.aten.reshape.default(sum_102, [512]);  sum_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_657: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_655, add_48)
    mul_658: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_655, rsqrt_13);  mul_655 = None
    sum_103: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True);  mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_304: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_300, mul_658);  add_300 = mul_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_125: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
    mul_659: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_103, -0.5);  sum_103 = None
    mul_660: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_659, pow_125);  mul_659 = pow_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_124: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_660, [1, 128, 512]);  mul_660 = None
    div_58: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_124, 512);  expand_124 = None
    pow_126: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 1.0);  add_48 = None
    mul_661: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_126, 2.0);  pow_126 = None
    mul_662: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_58, mul_661);  div_58 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_305: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_304, mul_662);  add_304 = mul_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_65: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_663: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_664: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_305, mul_663);  mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_963: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_664, [128, 512]);  mul_664 = None
    permute_817: "f32[512, 128]" = torch.ops.aten.permute.default(view_963, [1, 0])
    mm_343: "f32[512, 384]" = torch.ops.aten.mm.default(permute_817, view_175);  permute_817 = view_175 = None
    permute_818: "f32[384, 512]" = torch.ops.aten.permute.default(mm_343, [1, 0]);  mm_343 = None
    mm_344: "f32[128, 384]" = torch.ops.aten.mm.default(view_963, permute_819);  view_963 = permute_819 = None
    view_964: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_344, [1, 128, 384]);  mm_344 = None
    permute_820: "f32[512, 384]" = torch.ops.aten.permute.default(permute_818, [1, 0]);  permute_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_965: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_964, [1, 128, 6, 64]);  view_964 = None
    permute_821: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_965, [0, 2, 1, 3]);  view_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_966: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_821, [6, 128, 64]);  permute_821 = None
    bmm_116: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_822, view_966);  permute_822 = None
    bmm_117: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_966, permute_823);  view_966 = permute_823 = None
    view_967: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_116, [1, 6, 128, 64]);  bmm_116 = None
    view_968: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_117, [1, 6, 128, 128]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_66: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_665: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_666: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_968, mul_665);  view_968 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_667: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_666, alias_140);  mul_666 = None
    sum_104: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_667, [-1], True)
    mul_668: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_140, sum_104);  alias_140 = sum_104 = None
    sub_59: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_667, mul_668);  mul_667 = mul_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_19: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_59, 0);  sub_59 = None
    as_strided_scatter_34: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_19, [6, 128, 128], [16384, 128, 1], 0);  squeeze_19 = None
    as_strided_122: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_34, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_34 = None
    new_empty_strided_17: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_122, [6, 128, 128], [16384, 128, 1])
    as_strided_124: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_122, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_35: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_122, as_strided_124, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_306: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(as_strided_117, as_strided_124);  as_strided_117 = as_strided_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_118: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_824, as_strided_scatter_35);  permute_824 = None
    bmm_119: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_35, permute_825);  as_strided_scatter_35 = permute_825 = None
    view_969: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_118, [1, 6, 64, 128]);  bmm_118 = None
    view_970: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_119, [1, 6, 128, 64]);  bmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_826: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_969, [0, 1, 3, 2]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_827: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_967, [0, 2, 1, 3]);  view_967 = None
    clone_152: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_827, memory_format = torch.contiguous_format);  permute_827 = None
    view_971: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_152, [1, 128, 384]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_972: "f32[128, 384]" = torch.ops.aten.reshape.default(view_971, [128, 384]);  view_971 = None
    permute_828: "f32[384, 128]" = torch.ops.aten.permute.default(view_972, [1, 0])
    mm_345: "f32[384, 512]" = torch.ops.aten.mm.default(permute_828, view_157);  permute_828 = None
    permute_829: "f32[512, 384]" = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
    mm_346: "f32[128, 512]" = torch.ops.aten.mm.default(view_972, permute_830);  view_972 = permute_830 = None
    view_973: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_346, [1, 128, 512]);  mm_346 = None
    permute_831: "f32[384, 512]" = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_832: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_826, [0, 2, 1, 3]);  permute_826 = None
    view_974: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_832, [1, 128, 384]);  permute_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_975: "f32[128, 384]" = torch.ops.aten.reshape.default(view_974, [128, 384]);  view_974 = None
    permute_833: "f32[384, 128]" = torch.ops.aten.permute.default(view_975, [1, 0])
    mm_347: "f32[384, 512]" = torch.ops.aten.mm.default(permute_833, view_157);  permute_833 = None
    permute_834: "f32[512, 384]" = torch.ops.aten.permute.default(mm_347, [1, 0]);  mm_347 = None
    mm_348: "f32[128, 512]" = torch.ops.aten.mm.default(view_975, permute_835);  view_975 = permute_835 = None
    view_976: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_348, [1, 128, 512]);  mm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_307: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_973, view_976);  view_973 = view_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_836: "f32[384, 512]" = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_837: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
    clone_153: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_837, memory_format = torch.contiguous_format);  permute_837 = None
    view_977: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 384]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_978: "f32[128, 384]" = torch.ops.aten.reshape.default(view_977, [128, 384]);  view_977 = None
    permute_838: "f32[384, 128]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_349: "f32[384, 512]" = torch.ops.aten.mm.default(permute_838, view_157);  permute_838 = view_157 = None
    permute_839: "f32[512, 384]" = torch.ops.aten.permute.default(mm_349, [1, 0]);  mm_349 = None
    mm_350: "f32[128, 512]" = torch.ops.aten.mm.default(view_978, permute_840);  view_978 = permute_840 = None
    view_979: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_350, [1, 128, 512]);  mm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_308: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_307, view_979);  add_307 = view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_841: "f32[384, 512]" = torch.ops.aten.permute.default(permute_839, [1, 0]);  permute_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_669: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_308, primals_13);  primals_13 = None
    mul_670: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_308, mul_57);  add_308 = mul_57 = None
    sum_105: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 1], True);  mul_670 = None
    view_980: "f32[512]" = torch.ops.aten.reshape.default(sum_105, [512]);  sum_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_671: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_669, add_45)
    mul_672: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_669, rsqrt_12);  mul_669 = None
    sum_106: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_309: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_305, mul_672);  add_305 = mul_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_127: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
    mul_673: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_106, -0.5);  sum_106 = None
    mul_674: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_673, pow_127);  mul_673 = pow_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_125: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_674, [1, 128, 512]);  mul_674 = None
    div_59: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_125, 512);  expand_125 = None
    pow_128: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_45, 1.0);  add_45 = None
    mul_675: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_128, 2.0);  pow_128 = None
    mul_676: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_59, mul_675);  div_59 = mul_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_310: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_309, mul_676);  add_309 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_67: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_677: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_678: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_310, mul_677);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_981: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_678, [128, 512]);  mul_678 = None
    permute_842: "f32[512, 128]" = torch.ops.aten.permute.default(view_981, [1, 0])
    mm_351: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_842, view_155);  permute_842 = view_155 = None
    permute_843: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_351, [1, 0]);  mm_351 = None
    mm_352: "f32[128, 1024]" = torch.ops.aten.mm.default(view_981, permute_844);  view_981 = permute_844 = None
    view_982: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_352, [1, 128, 1024]);  mm_352 = None
    permute_845: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_679: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_680: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_982, mul_679);  view_982 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_681: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_680, mul_55);  mul_55 = None
    mul_682: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_680, view_154);  mul_680 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_983: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_681, [128, 1024]);  mul_681 = None
    permute_846: "f32[1024, 128]" = torch.ops.aten.permute.default(view_983, [1, 0])
    mm_353: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_846, view_151);  permute_846 = None
    permute_847: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_353, [1, 0]);  mm_353 = None
    mm_354: "f32[128, 512]" = torch.ops.aten.mm.default(view_983, permute_848);  view_983 = permute_848 = None
    view_984: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_354, [1, 128, 512]);  mm_354 = None
    permute_849: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_683: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_682, mul_52);  mul_52 = None
    mul_684: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_682, add_44);  mul_682 = add_44 = None
    mul_685: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
    sub_60: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_685);  mul_685 = None
    mul_686: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_683, sub_60);  mul_683 = sub_60 = None
    mul_687: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_686, 0.7978845608028654);  mul_686 = None
    mul_688: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_687, 0.044715)
    pow_129: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_152, 2.0);  view_152 = None
    mul_689: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_129, 3.0);  pow_129 = None
    mul_690: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_311: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_687, mul_690);  mul_687 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_691: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_684, 0.5);  mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_312: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_311, mul_691);  add_311 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_985: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_312, [128, 1024]);  add_312 = None
    permute_850: "f32[1024, 128]" = torch.ops.aten.permute.default(view_985, [1, 0])
    mm_355: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_850, view_151);  permute_850 = view_151 = None
    permute_851: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_355, [1, 0]);  mm_355 = None
    mm_356: "f32[128, 512]" = torch.ops.aten.mm.default(view_985, permute_852);  view_985 = permute_852 = None
    view_986: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_356, [1, 128, 512]);  mm_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_313: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_984, view_986);  view_984 = view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_853: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_851, [1, 0]);  permute_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_692: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_313, primals_12);  primals_12 = None
    mul_693: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_313, mul_50);  add_313 = mul_50 = None
    sum_107: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 1], True);  mul_693 = None
    view_987: "f32[512]" = torch.ops.aten.reshape.default(sum_107, [512]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_694: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_692, add_41)
    mul_695: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_692, rsqrt_11);  mul_692 = None
    sum_108: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [2], True);  mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_314: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_310, mul_695);  add_310 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_130: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
    mul_696: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_108, -0.5);  sum_108 = None
    mul_697: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_696, pow_130);  mul_696 = pow_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_126: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_697, [1, 128, 512]);  mul_697 = None
    div_60: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_126, 512);  expand_126 = None
    pow_131: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 1.0);  add_41 = None
    mul_698: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_131, 2.0);  pow_131 = None
    mul_699: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_60, mul_698);  div_60 = mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_315: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_314, mul_699);  add_314 = mul_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_69: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_700: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_701: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_315, mul_700);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_988: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_701, [128, 512]);  mul_701 = None
    permute_854: "f32[512, 128]" = torch.ops.aten.permute.default(view_988, [1, 0])
    mm_357: "f32[512, 384]" = torch.ops.aten.mm.default(permute_854, view_149);  permute_854 = view_149 = None
    permute_855: "f32[384, 512]" = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
    mm_358: "f32[128, 384]" = torch.ops.aten.mm.default(view_988, permute_856);  view_988 = permute_856 = None
    view_989: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_358, [1, 128, 384]);  mm_358 = None
    permute_857: "f32[512, 384]" = torch.ops.aten.permute.default(permute_855, [1, 0]);  permute_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_990: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_989, [1, 128, 6, 64]);  view_989 = None
    permute_858: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_990, [0, 2, 1, 3]);  view_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_991: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_858, [6, 128, 64]);  permute_858 = None
    bmm_120: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_859, view_991);  permute_859 = None
    bmm_121: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_991, permute_860);  view_991 = permute_860 = None
    view_992: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_120, [1, 6, 128, 64]);  bmm_120 = None
    view_993: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_121, [1, 6, 128, 128]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_70: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_702: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_703: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_993, mul_702);  view_993 = mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_704: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_703, alias_144);  mul_703 = None
    sum_109: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [-1], True)
    mul_705: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_144, sum_109);  alias_144 = sum_109 = None
    sub_61: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_20: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_61, 0);  sub_61 = None
    as_strided_scatter_36: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_20, [6, 128, 128], [16384, 128, 1], 0);  squeeze_20 = None
    as_strided_129: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_36, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_36 = None
    new_empty_strided_18: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_129, [6, 128, 128], [16384, 128, 1])
    as_strided_131: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_129, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_37: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_129, as_strided_131, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_316: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_306, as_strided_131);  add_306 = as_strided_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_122: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_861, as_strided_scatter_37);  permute_861 = None
    bmm_123: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_37, permute_862);  as_strided_scatter_37 = permute_862 = None
    view_994: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_122, [1, 6, 64, 128]);  bmm_122 = None
    view_995: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_123, [1, 6, 128, 64]);  bmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_863: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_994, [0, 1, 3, 2]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_864: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_992, [0, 2, 1, 3]);  view_992 = None
    clone_159: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
    view_996: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_159, [1, 128, 384]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_997: "f32[128, 384]" = torch.ops.aten.reshape.default(view_996, [128, 384]);  view_996 = None
    permute_865: "f32[384, 128]" = torch.ops.aten.permute.default(view_997, [1, 0])
    mm_359: "f32[384, 512]" = torch.ops.aten.mm.default(permute_865, view_131);  permute_865 = None
    permute_866: "f32[512, 384]" = torch.ops.aten.permute.default(mm_359, [1, 0]);  mm_359 = None
    mm_360: "f32[128, 512]" = torch.ops.aten.mm.default(view_997, permute_867);  view_997 = permute_867 = None
    view_998: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_360, [1, 128, 512]);  mm_360 = None
    permute_868: "f32[384, 512]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_869: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_863, [0, 2, 1, 3]);  permute_863 = None
    view_999: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_869, [1, 128, 384]);  permute_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1000: "f32[128, 384]" = torch.ops.aten.reshape.default(view_999, [128, 384]);  view_999 = None
    permute_870: "f32[384, 128]" = torch.ops.aten.permute.default(view_1000, [1, 0])
    mm_361: "f32[384, 512]" = torch.ops.aten.mm.default(permute_870, view_131);  permute_870 = None
    permute_871: "f32[512, 384]" = torch.ops.aten.permute.default(mm_361, [1, 0]);  mm_361 = None
    mm_362: "f32[128, 512]" = torch.ops.aten.mm.default(view_1000, permute_872);  view_1000 = permute_872 = None
    view_1001: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_362, [1, 128, 512]);  mm_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_317: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_998, view_1001);  view_998 = view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_873: "f32[384, 512]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_874: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_995, [0, 2, 1, 3]);  view_995 = None
    clone_160: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_874, memory_format = torch.contiguous_format);  permute_874 = None
    view_1002: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_160, [1, 128, 384]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1003: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1002, [128, 384]);  view_1002 = None
    permute_875: "f32[384, 128]" = torch.ops.aten.permute.default(view_1003, [1, 0])
    mm_363: "f32[384, 512]" = torch.ops.aten.mm.default(permute_875, view_131);  permute_875 = view_131 = None
    permute_876: "f32[512, 384]" = torch.ops.aten.permute.default(mm_363, [1, 0]);  mm_363 = None
    mm_364: "f32[128, 512]" = torch.ops.aten.mm.default(view_1003, permute_877);  view_1003 = permute_877 = None
    view_1004: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_364, [1, 128, 512]);  mm_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_318: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_317, view_1004);  add_317 = view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_878: "f32[384, 512]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_706: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_318, primals_11);  primals_11 = None
    mul_707: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_318, mul_48);  add_318 = mul_48 = None
    sum_110: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1], True);  mul_707 = None
    view_1005: "f32[512]" = torch.ops.aten.reshape.default(sum_110, [512]);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_708: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_706, add_38)
    mul_709: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_706, rsqrt_10);  mul_706 = None
    sum_111: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [2], True);  mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_319: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_315, mul_709);  add_315 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_132: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
    mul_710: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_111, -0.5);  sum_111 = None
    mul_711: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_710, pow_132);  mul_710 = pow_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_127: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_711, [1, 128, 512]);  mul_711 = None
    div_61: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_127, 512);  expand_127 = None
    pow_133: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_38, 1.0);  add_38 = None
    mul_712: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_133, 2.0);  pow_133 = None
    mul_713: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_61, mul_712);  div_61 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_320: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_319, mul_713);  add_319 = mul_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_71: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_714: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_715: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_320, mul_714);  mul_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_1006: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_715, [128, 512]);  mul_715 = None
    permute_879: "f32[512, 128]" = torch.ops.aten.permute.default(view_1006, [1, 0])
    mm_365: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_879, view_129);  permute_879 = view_129 = None
    permute_880: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_365, [1, 0]);  mm_365 = None
    mm_366: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1006, permute_881);  view_1006 = permute_881 = None
    view_1007: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_366, [1, 128, 1024]);  mm_366 = None
    permute_882: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_880, [1, 0]);  permute_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_72: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_39, torch.float32);  getitem_39 = None
    mul_716: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
    mul_717: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1007, mul_716);  view_1007 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_718: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_717, mul_46);  mul_46 = None
    mul_719: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_717, view_128);  mul_717 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_1008: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_718, [128, 1024]);  mul_718 = None
    permute_883: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1008, [1, 0])
    mm_367: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_883, view_125);  permute_883 = None
    permute_884: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_367, [1, 0]);  mm_367 = None
    mm_368: "f32[128, 512]" = torch.ops.aten.mm.default(view_1008, permute_885);  view_1008 = permute_885 = None
    view_1009: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_368, [1, 128, 512]);  mm_368 = None
    permute_886: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_884, [1, 0]);  permute_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_720: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_719, mul_43);  mul_43 = None
    mul_721: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_719, add_37);  mul_719 = add_37 = None
    mul_722: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
    sub_62: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_722);  mul_722 = None
    mul_723: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_720, sub_62);  mul_720 = sub_62 = None
    mul_724: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_723, 0.7978845608028654);  mul_723 = None
    mul_725: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_724, 0.044715)
    pow_134: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_126, 2.0);  view_126 = None
    mul_726: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_134, 3.0);  pow_134 = None
    mul_727: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_321: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_724, mul_727);  mul_724 = mul_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_728: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_721, 0.5);  mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_322: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_321, mul_728);  add_321 = mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_1010: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_322, [128, 1024]);  add_322 = None
    permute_887: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1010, [1, 0])
    mm_369: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_887, view_125);  permute_887 = view_125 = None
    permute_888: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_369, [1, 0]);  mm_369 = None
    mm_370: "f32[128, 512]" = torch.ops.aten.mm.default(view_1010, permute_889);  view_1010 = permute_889 = None
    view_1011: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_370, [1, 128, 512]);  mm_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_323: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1009, view_1011);  view_1009 = view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_890: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_729: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_323, primals_10);  primals_10 = None
    mul_730: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_323, mul_41);  add_323 = mul_41 = None
    sum_112: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 1], True);  mul_730 = None
    view_1012: "f32[512]" = torch.ops.aten.reshape.default(sum_112, [512]);  sum_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_731: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_729, add_34)
    mul_732: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_729, rsqrt_9);  mul_729 = None
    sum_113: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True);  mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_324: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_320, mul_732);  add_320 = mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_135: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
    mul_733: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_113, -0.5);  sum_113 = None
    mul_734: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_733, pow_135);  mul_733 = pow_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_128: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_734, [1, 128, 512]);  mul_734 = None
    div_62: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_128, 512);  expand_128 = None
    pow_136: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0);  add_34 = None
    mul_735: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_136, 2.0);  pow_136 = None
    mul_736: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_62, mul_735);  div_62 = mul_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_325: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_324, mul_736);  add_324 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_73: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_737: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
    mul_738: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_325, mul_737);  mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_1013: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_738, [128, 512]);  mul_738 = None
    permute_891: "f32[512, 128]" = torch.ops.aten.permute.default(view_1013, [1, 0])
    mm_371: "f32[512, 384]" = torch.ops.aten.mm.default(permute_891, view_123);  permute_891 = view_123 = None
    permute_892: "f32[384, 512]" = torch.ops.aten.permute.default(mm_371, [1, 0]);  mm_371 = None
    mm_372: "f32[128, 384]" = torch.ops.aten.mm.default(view_1013, permute_893);  view_1013 = permute_893 = None
    view_1014: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_372, [1, 128, 384]);  mm_372 = None
    permute_894: "f32[512, 384]" = torch.ops.aten.permute.default(permute_892, [1, 0]);  permute_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_1015: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1014, [1, 128, 6, 64]);  view_1014 = None
    permute_895: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1015, [0, 2, 1, 3]);  view_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_1016: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_895, [6, 128, 64]);  permute_895 = None
    bmm_124: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_896, view_1016);  permute_896 = None
    bmm_125: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_1016, permute_897);  view_1016 = permute_897 = None
    view_1017: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_124, [1, 6, 128, 64]);  bmm_124 = None
    view_1018: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_125, [1, 6, 128, 128]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_74: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_739: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
    mul_740: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1018, mul_739);  view_1018 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_741: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_740, alias_148);  mul_740 = None
    sum_114: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [-1], True)
    mul_742: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_148, sum_114);  alias_148 = sum_114 = None
    sub_63: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_21: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_63, 0);  sub_63 = None
    as_strided_scatter_38: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_21, [6, 128, 128], [16384, 128, 1], 0);  squeeze_21 = None
    as_strided_136: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_38, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_38 = None
    new_empty_strided_19: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_136, [6, 128, 128], [16384, 128, 1])
    as_strided_138: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_136, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_39: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_136, as_strided_138, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_326: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_316, as_strided_138);  add_316 = as_strided_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_126: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_898, as_strided_scatter_39);  permute_898 = None
    bmm_127: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_39, permute_899);  as_strided_scatter_39 = permute_899 = None
    view_1019: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_126, [1, 6, 64, 128]);  bmm_126 = None
    view_1020: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_127, [1, 6, 128, 64]);  bmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_900: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1019, [0, 1, 3, 2]);  view_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_901: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1017, [0, 2, 1, 3]);  view_1017 = None
    clone_166: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_901, memory_format = torch.contiguous_format);  permute_901 = None
    view_1021: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_166, [1, 128, 384]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1022: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1021, [128, 384]);  view_1021 = None
    permute_902: "f32[384, 128]" = torch.ops.aten.permute.default(view_1022, [1, 0])
    mm_373: "f32[384, 512]" = torch.ops.aten.mm.default(permute_902, view_105);  permute_902 = None
    permute_903: "f32[512, 384]" = torch.ops.aten.permute.default(mm_373, [1, 0]);  mm_373 = None
    mm_374: "f32[128, 512]" = torch.ops.aten.mm.default(view_1022, permute_904);  view_1022 = permute_904 = None
    view_1023: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_374, [1, 128, 512]);  mm_374 = None
    permute_905: "f32[384, 512]" = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_906: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_900, [0, 2, 1, 3]);  permute_900 = None
    view_1024: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_906, [1, 128, 384]);  permute_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1025: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1024, [128, 384]);  view_1024 = None
    permute_907: "f32[384, 128]" = torch.ops.aten.permute.default(view_1025, [1, 0])
    mm_375: "f32[384, 512]" = torch.ops.aten.mm.default(permute_907, view_105);  permute_907 = None
    permute_908: "f32[512, 384]" = torch.ops.aten.permute.default(mm_375, [1, 0]);  mm_375 = None
    mm_376: "f32[128, 512]" = torch.ops.aten.mm.default(view_1025, permute_909);  view_1025 = permute_909 = None
    view_1026: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_376, [1, 128, 512]);  mm_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_327: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1023, view_1026);  view_1023 = view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_910: "f32[384, 512]" = torch.ops.aten.permute.default(permute_908, [1, 0]);  permute_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_911: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1020, [0, 2, 1, 3]);  view_1020 = None
    clone_167: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_911, memory_format = torch.contiguous_format);  permute_911 = None
    view_1027: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_167, [1, 128, 384]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1028: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1027, [128, 384]);  view_1027 = None
    permute_912: "f32[384, 128]" = torch.ops.aten.permute.default(view_1028, [1, 0])
    mm_377: "f32[384, 512]" = torch.ops.aten.mm.default(permute_912, view_105);  permute_912 = view_105 = None
    permute_913: "f32[512, 384]" = torch.ops.aten.permute.default(mm_377, [1, 0]);  mm_377 = None
    mm_378: "f32[128, 512]" = torch.ops.aten.mm.default(view_1028, permute_914);  view_1028 = permute_914 = None
    view_1029: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_378, [1, 128, 512]);  mm_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_328: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_327, view_1029);  add_327 = view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_915: "f32[384, 512]" = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_743: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_328, primals_9);  primals_9 = None
    mul_744: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_328, mul_39);  add_328 = mul_39 = None
    sum_115: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 1], True);  mul_744 = None
    view_1030: "f32[512]" = torch.ops.aten.reshape.default(sum_115, [512]);  sum_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_745: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_743, add_31)
    mul_746: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_743, rsqrt_8);  mul_743 = None
    sum_116: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [2], True);  mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_329: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_325, mul_746);  add_325 = mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_137: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
    mul_747: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_116, -0.5);  sum_116 = None
    mul_748: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_747, pow_137);  mul_747 = pow_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_129: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_748, [1, 128, 512]);  mul_748 = None
    div_63: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_129, 512);  expand_129 = None
    pow_138: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
    mul_749: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_138, 2.0);  pow_138 = None
    mul_750: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_63, mul_749);  div_63 = mul_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_330: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_329, mul_750);  add_329 = mul_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_75: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_751: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
    mul_752: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_330, mul_751);  mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_1031: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_752, [128, 512]);  mul_752 = None
    permute_916: "f32[512, 128]" = torch.ops.aten.permute.default(view_1031, [1, 0])
    mm_379: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_916, view_103);  permute_916 = view_103 = None
    permute_917: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_379, [1, 0]);  mm_379 = None
    mm_380: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1031, permute_918);  view_1031 = permute_918 = None
    view_1032: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_380, [1, 128, 1024]);  mm_380 = None
    permute_919: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_917, [1, 0]);  permute_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_76: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_753: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
    mul_754: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1032, mul_753);  view_1032 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_755: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_754, mul_37);  mul_37 = None
    mul_756: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_754, view_102);  mul_754 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_1033: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_755, [128, 1024]);  mul_755 = None
    permute_920: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1033, [1, 0])
    mm_381: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_920, view_99);  permute_920 = None
    permute_921: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_381, [1, 0]);  mm_381 = None
    mm_382: "f32[128, 512]" = torch.ops.aten.mm.default(view_1033, permute_922);  view_1033 = permute_922 = None
    view_1034: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_382, [1, 128, 512]);  mm_382 = None
    permute_923: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_921, [1, 0]);  permute_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_757: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_756, mul_34);  mul_34 = None
    mul_758: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_756, add_30);  mul_756 = add_30 = None
    mul_759: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
    sub_64: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_759);  mul_759 = None
    mul_760: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_757, sub_64);  mul_757 = sub_64 = None
    mul_761: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_760, 0.7978845608028654);  mul_760 = None
    mul_762: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_761, 0.044715)
    pow_139: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_100, 2.0);  view_100 = None
    mul_763: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_139, 3.0);  pow_139 = None
    mul_764: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_331: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_761, mul_764);  mul_761 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_765: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_758, 0.5);  mul_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_332: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_331, mul_765);  add_331 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_1035: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_332, [128, 1024]);  add_332 = None
    permute_924: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1035, [1, 0])
    mm_383: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_924, view_99);  permute_924 = view_99 = None
    permute_925: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_383, [1, 0]);  mm_383 = None
    mm_384: "f32[128, 512]" = torch.ops.aten.mm.default(view_1035, permute_926);  view_1035 = permute_926 = None
    view_1036: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_384, [1, 128, 512]);  mm_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_333: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1034, view_1036);  view_1034 = view_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_927: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_925, [1, 0]);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_766: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_333, primals_8);  primals_8 = None
    mul_767: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_333, mul_32);  add_333 = mul_32 = None
    sum_117: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 1], True);  mul_767 = None
    view_1037: "f32[512]" = torch.ops.aten.reshape.default(sum_117, [512]);  sum_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_768: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_766, add_27)
    mul_769: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_766, rsqrt_7);  mul_766 = None
    sum_118: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True);  mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_334: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_330, mul_769);  add_330 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_140: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
    mul_770: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_118, -0.5);  sum_118 = None
    mul_771: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_770, pow_140);  mul_770 = pow_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_130: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_771, [1, 128, 512]);  mul_771 = None
    div_64: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_130, 512);  expand_130 = None
    pow_141: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 1.0);  add_27 = None
    mul_772: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_141, 2.0);  pow_141 = None
    mul_773: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_64, mul_772);  div_64 = mul_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_335: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_334, mul_773);  add_334 = mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_77: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_774: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
    mul_775: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_335, mul_774);  mul_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_1038: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_775, [128, 512]);  mul_775 = None
    permute_928: "f32[512, 128]" = torch.ops.aten.permute.default(view_1038, [1, 0])
    mm_385: "f32[512, 384]" = torch.ops.aten.mm.default(permute_928, view_97);  permute_928 = view_97 = None
    permute_929: "f32[384, 512]" = torch.ops.aten.permute.default(mm_385, [1, 0]);  mm_385 = None
    mm_386: "f32[128, 384]" = torch.ops.aten.mm.default(view_1038, permute_930);  view_1038 = permute_930 = None
    view_1039: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_386, [1, 128, 384]);  mm_386 = None
    permute_931: "f32[512, 384]" = torch.ops.aten.permute.default(permute_929, [1, 0]);  permute_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_1040: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1039, [1, 128, 6, 64]);  view_1039 = None
    permute_932: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1040, [0, 2, 1, 3]);  view_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_1041: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_932, [6, 128, 64]);  permute_932 = None
    bmm_128: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_933, view_1041);  permute_933 = None
    bmm_129: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_1041, permute_934);  view_1041 = permute_934 = None
    view_1042: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_128, [1, 6, 128, 64]);  bmm_128 = None
    view_1043: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_129, [1, 6, 128, 128]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_78: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_776: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
    mul_777: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1043, mul_776);  view_1043 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_778: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_777, alias_152);  mul_777 = None
    sum_119: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [-1], True)
    mul_779: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_152, sum_119);  alias_152 = sum_119 = None
    sub_65: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_22: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_65, 0);  sub_65 = None
    as_strided_scatter_40: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_22, [6, 128, 128], [16384, 128, 1], 0);  squeeze_22 = None
    as_strided_143: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_40, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_40 = None
    new_empty_strided_20: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_143, [6, 128, 128], [16384, 128, 1])
    as_strided_145: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_143, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_41: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_143, as_strided_145, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_336: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_326, as_strided_145);  add_326 = as_strided_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_130: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_935, as_strided_scatter_41);  permute_935 = None
    bmm_131: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_41, permute_936);  as_strided_scatter_41 = permute_936 = None
    view_1044: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_130, [1, 6, 64, 128]);  bmm_130 = None
    view_1045: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_131, [1, 6, 128, 64]);  bmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_937: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1044, [0, 1, 3, 2]);  view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_938: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1042, [0, 2, 1, 3]);  view_1042 = None
    clone_173: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
    view_1046: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_173, [1, 128, 384]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1047: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1046, [128, 384]);  view_1046 = None
    permute_939: "f32[384, 128]" = torch.ops.aten.permute.default(view_1047, [1, 0])
    mm_387: "f32[384, 512]" = torch.ops.aten.mm.default(permute_939, view_79);  permute_939 = None
    permute_940: "f32[512, 384]" = torch.ops.aten.permute.default(mm_387, [1, 0]);  mm_387 = None
    mm_388: "f32[128, 512]" = torch.ops.aten.mm.default(view_1047, permute_941);  view_1047 = permute_941 = None
    view_1048: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_388, [1, 128, 512]);  mm_388 = None
    permute_942: "f32[384, 512]" = torch.ops.aten.permute.default(permute_940, [1, 0]);  permute_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_943: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_937, [0, 2, 1, 3]);  permute_937 = None
    view_1049: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_943, [1, 128, 384]);  permute_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1050: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1049, [128, 384]);  view_1049 = None
    permute_944: "f32[384, 128]" = torch.ops.aten.permute.default(view_1050, [1, 0])
    mm_389: "f32[384, 512]" = torch.ops.aten.mm.default(permute_944, view_79);  permute_944 = None
    permute_945: "f32[512, 384]" = torch.ops.aten.permute.default(mm_389, [1, 0]);  mm_389 = None
    mm_390: "f32[128, 512]" = torch.ops.aten.mm.default(view_1050, permute_946);  view_1050 = permute_946 = None
    view_1051: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_390, [1, 128, 512]);  mm_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_337: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1048, view_1051);  view_1048 = view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_947: "f32[384, 512]" = torch.ops.aten.permute.default(permute_945, [1, 0]);  permute_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_948: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1045, [0, 2, 1, 3]);  view_1045 = None
    clone_174: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
    view_1052: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_174, [1, 128, 384]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1053: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1052, [128, 384]);  view_1052 = None
    permute_949: "f32[384, 128]" = torch.ops.aten.permute.default(view_1053, [1, 0])
    mm_391: "f32[384, 512]" = torch.ops.aten.mm.default(permute_949, view_79);  permute_949 = view_79 = None
    permute_950: "f32[512, 384]" = torch.ops.aten.permute.default(mm_391, [1, 0]);  mm_391 = None
    mm_392: "f32[128, 512]" = torch.ops.aten.mm.default(view_1053, permute_951);  view_1053 = permute_951 = None
    view_1054: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_392, [1, 128, 512]);  mm_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_338: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_337, view_1054);  add_337 = view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_952: "f32[384, 512]" = torch.ops.aten.permute.default(permute_950, [1, 0]);  permute_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_780: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_338, primals_7);  primals_7 = None
    mul_781: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_338, mul_30);  add_338 = mul_30 = None
    sum_120: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 1], True);  mul_781 = None
    view_1055: "f32[512]" = torch.ops.aten.reshape.default(sum_120, [512]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_782: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_780, add_24)
    mul_783: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_780, rsqrt_6);  mul_780 = None
    sum_121: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [2], True);  mul_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_339: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_335, mul_783);  add_335 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_142: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
    mul_784: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_121, -0.5);  sum_121 = None
    mul_785: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_784, pow_142);  mul_784 = pow_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_131: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_785, [1, 128, 512]);  mul_785 = None
    div_65: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_131, 512);  expand_131 = None
    pow_143: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 1.0);  add_24 = None
    mul_786: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_143, 2.0);  pow_143 = None
    mul_787: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_65, mul_786);  div_65 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_340: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_339, mul_787);  add_339 = mul_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_79: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_788: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
    mul_789: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_340, mul_788);  mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_1056: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_789, [128, 512]);  mul_789 = None
    permute_953: "f32[512, 128]" = torch.ops.aten.permute.default(view_1056, [1, 0])
    mm_393: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_953, view_77);  permute_953 = view_77 = None
    permute_954: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_393, [1, 0]);  mm_393 = None
    mm_394: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1056, permute_955);  view_1056 = permute_955 = None
    view_1057: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_394, [1, 128, 1024]);  mm_394 = None
    permute_956: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_954, [1, 0]);  permute_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_80: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_790: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
    mul_791: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1057, mul_790);  view_1057 = mul_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_792: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_791, mul_28);  mul_28 = None
    mul_793: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_791, view_76);  mul_791 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_1058: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_792, [128, 1024]);  mul_792 = None
    permute_957: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1058, [1, 0])
    mm_395: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_957, view_73);  permute_957 = None
    permute_958: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_395, [1, 0]);  mm_395 = None
    mm_396: "f32[128, 512]" = torch.ops.aten.mm.default(view_1058, permute_959);  view_1058 = permute_959 = None
    view_1059: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_396, [1, 128, 512]);  mm_396 = None
    permute_960: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_958, [1, 0]);  permute_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_794: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_793, mul_25);  mul_25 = None
    mul_795: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_793, add_23);  mul_793 = add_23 = None
    mul_796: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
    sub_66: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_796);  mul_796 = None
    mul_797: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_794, sub_66);  mul_794 = sub_66 = None
    mul_798: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_797, 0.7978845608028654);  mul_797 = None
    mul_799: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_798, 0.044715)
    pow_144: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_74, 2.0);  view_74 = None
    mul_800: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_144, 3.0);  pow_144 = None
    mul_801: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_341: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_798, mul_801);  mul_798 = mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_802: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_795, 0.5);  mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_342: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_341, mul_802);  add_341 = mul_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_1060: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_342, [128, 1024]);  add_342 = None
    permute_961: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1060, [1, 0])
    mm_397: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_961, view_73);  permute_961 = view_73 = None
    permute_962: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_397, [1, 0]);  mm_397 = None
    mm_398: "f32[128, 512]" = torch.ops.aten.mm.default(view_1060, permute_963);  view_1060 = permute_963 = None
    view_1061: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_398, [1, 128, 512]);  mm_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_343: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1059, view_1061);  view_1059 = view_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_964: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_962, [1, 0]);  permute_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_803: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_343, primals_6);  primals_6 = None
    mul_804: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_343, mul_23);  add_343 = mul_23 = None
    sum_122: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_804, [0, 1], True);  mul_804 = None
    view_1062: "f32[512]" = torch.ops.aten.reshape.default(sum_122, [512]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_805: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_803, add_20)
    mul_806: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_803, rsqrt_5);  mul_803 = None
    sum_123: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_344: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_340, mul_806);  add_340 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_145: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
    mul_807: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_123, -0.5);  sum_123 = None
    mul_808: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_807, pow_145);  mul_807 = pow_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_132: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_808, [1, 128, 512]);  mul_808 = None
    div_66: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_132, 512);  expand_132 = None
    pow_146: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 1.0);  add_20 = None
    mul_809: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_146, 2.0);  pow_146 = None
    mul_810: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_66, mul_809);  div_66 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_345: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_344, mul_810);  add_344 = mul_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_81: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_811: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
    mul_812: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_345, mul_811);  mul_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_1063: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_812, [128, 512]);  mul_812 = None
    permute_965: "f32[512, 128]" = torch.ops.aten.permute.default(view_1063, [1, 0])
    mm_399: "f32[512, 384]" = torch.ops.aten.mm.default(permute_965, view_71);  permute_965 = view_71 = None
    permute_966: "f32[384, 512]" = torch.ops.aten.permute.default(mm_399, [1, 0]);  mm_399 = None
    mm_400: "f32[128, 384]" = torch.ops.aten.mm.default(view_1063, permute_967);  view_1063 = permute_967 = None
    view_1064: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_400, [1, 128, 384]);  mm_400 = None
    permute_968: "f32[512, 384]" = torch.ops.aten.permute.default(permute_966, [1, 0]);  permute_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_1065: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1064, [1, 128, 6, 64]);  view_1064 = None
    permute_969: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1065, [0, 2, 1, 3]);  view_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_1066: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_969, [6, 128, 64]);  permute_969 = None
    bmm_132: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_970, view_1066);  permute_970 = None
    bmm_133: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_1066, permute_971);  view_1066 = permute_971 = None
    view_1067: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_132, [1, 6, 128, 64]);  bmm_132 = None
    view_1068: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_133, [1, 6, 128, 128]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_82: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_813: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
    mul_814: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1068, mul_813);  view_1068 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_815: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_814, alias_156);  mul_814 = None
    sum_124: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [-1], True)
    mul_816: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_156, sum_124);  alias_156 = sum_124 = None
    sub_67: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_815, mul_816);  mul_815 = mul_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_23: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_67, 0);  sub_67 = None
    as_strided_scatter_42: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_23, [6, 128, 128], [16384, 128, 1], 0);  squeeze_23 = None
    as_strided_150: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_42, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_42 = None
    new_empty_strided_21: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_150, [6, 128, 128], [16384, 128, 1])
    as_strided_152: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_150, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_43: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_150, as_strided_152, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_346: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_336, as_strided_152);  add_336 = as_strided_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_134: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_972, as_strided_scatter_43);  permute_972 = None
    bmm_135: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_43, permute_973);  as_strided_scatter_43 = permute_973 = None
    view_1069: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_134, [1, 6, 64, 128]);  bmm_134 = None
    view_1070: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_135, [1, 6, 128, 64]);  bmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_974: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1069, [0, 1, 3, 2]);  view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_975: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1067, [0, 2, 1, 3]);  view_1067 = None
    clone_180: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_975, memory_format = torch.contiguous_format);  permute_975 = None
    view_1071: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_180, [1, 128, 384]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1072: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1071, [128, 384]);  view_1071 = None
    permute_976: "f32[384, 128]" = torch.ops.aten.permute.default(view_1072, [1, 0])
    mm_401: "f32[384, 512]" = torch.ops.aten.mm.default(permute_976, view_53);  permute_976 = None
    permute_977: "f32[512, 384]" = torch.ops.aten.permute.default(mm_401, [1, 0]);  mm_401 = None
    mm_402: "f32[128, 512]" = torch.ops.aten.mm.default(view_1072, permute_978);  view_1072 = permute_978 = None
    view_1073: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_402, [1, 128, 512]);  mm_402 = None
    permute_979: "f32[384, 512]" = torch.ops.aten.permute.default(permute_977, [1, 0]);  permute_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_980: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_974, [0, 2, 1, 3]);  permute_974 = None
    view_1074: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_980, [1, 128, 384]);  permute_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1075: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1074, [128, 384]);  view_1074 = None
    permute_981: "f32[384, 128]" = torch.ops.aten.permute.default(view_1075, [1, 0])
    mm_403: "f32[384, 512]" = torch.ops.aten.mm.default(permute_981, view_53);  permute_981 = None
    permute_982: "f32[512, 384]" = torch.ops.aten.permute.default(mm_403, [1, 0]);  mm_403 = None
    mm_404: "f32[128, 512]" = torch.ops.aten.mm.default(view_1075, permute_983);  view_1075 = permute_983 = None
    view_1076: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_404, [1, 128, 512]);  mm_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_347: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1073, view_1076);  view_1073 = view_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_984: "f32[384, 512]" = torch.ops.aten.permute.default(permute_982, [1, 0]);  permute_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_985: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1070, [0, 2, 1, 3]);  view_1070 = None
    clone_181: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
    view_1077: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_181, [1, 128, 384]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1078: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1077, [128, 384]);  view_1077 = None
    permute_986: "f32[384, 128]" = torch.ops.aten.permute.default(view_1078, [1, 0])
    mm_405: "f32[384, 512]" = torch.ops.aten.mm.default(permute_986, view_53);  permute_986 = view_53 = None
    permute_987: "f32[512, 384]" = torch.ops.aten.permute.default(mm_405, [1, 0]);  mm_405 = None
    mm_406: "f32[128, 512]" = torch.ops.aten.mm.default(view_1078, permute_988);  view_1078 = permute_988 = None
    view_1079: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_406, [1, 128, 512]);  mm_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_348: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_347, view_1079);  add_347 = view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_989: "f32[384, 512]" = torch.ops.aten.permute.default(permute_987, [1, 0]);  permute_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_817: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_348, primals_5);  primals_5 = None
    mul_818: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_348, mul_21);  add_348 = mul_21 = None
    sum_125: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1], True);  mul_818 = None
    view_1080: "f32[512]" = torch.ops.aten.reshape.default(sum_125, [512]);  sum_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_819: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_817, add_17)
    mul_820: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_817, rsqrt_4);  mul_817 = None
    sum_126: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_819, [2], True);  mul_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_349: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_345, mul_820);  add_345 = mul_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_147: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
    mul_821: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_126, -0.5);  sum_126 = None
    mul_822: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_821, pow_147);  mul_821 = pow_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_133: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_822, [1, 128, 512]);  mul_822 = None
    div_67: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_133, 512);  expand_133 = None
    pow_148: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_17, 1.0);  add_17 = None
    mul_823: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_148, 2.0);  pow_148 = None
    mul_824: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_67, mul_823);  div_67 = mul_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_350: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_349, mul_824);  add_349 = mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_83: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_825: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
    mul_826: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_350, mul_825);  mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_1081: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_826, [128, 512]);  mul_826 = None
    permute_990: "f32[512, 128]" = torch.ops.aten.permute.default(view_1081, [1, 0])
    mm_407: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_990, view_51);  permute_990 = view_51 = None
    permute_991: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_407, [1, 0]);  mm_407 = None
    mm_408: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1081, permute_992);  view_1081 = permute_992 = None
    view_1082: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_408, [1, 128, 1024]);  mm_408 = None
    permute_993: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_991, [1, 0]);  permute_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_84: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_827: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
    mul_828: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1082, mul_827);  view_1082 = mul_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_829: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_828, mul_19);  mul_19 = None
    mul_830: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_828, view_50);  mul_828 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_1083: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_829, [128, 1024]);  mul_829 = None
    permute_994: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1083, [1, 0])
    mm_409: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_994, view_47);  permute_994 = None
    permute_995: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_409, [1, 0]);  mm_409 = None
    mm_410: "f32[128, 512]" = torch.ops.aten.mm.default(view_1083, permute_996);  view_1083 = permute_996 = None
    view_1084: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_410, [1, 128, 512]);  mm_410 = None
    permute_997: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_995, [1, 0]);  permute_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_831: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_830, mul_16);  mul_16 = None
    mul_832: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_830, add_16);  mul_830 = add_16 = None
    mul_833: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
    sub_68: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_833);  mul_833 = None
    mul_834: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_831, sub_68);  mul_831 = sub_68 = None
    mul_835: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_834, 0.7978845608028654);  mul_834 = None
    mul_836: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_835, 0.044715)
    pow_149: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_48, 2.0);  view_48 = None
    mul_837: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_149, 3.0);  pow_149 = None
    mul_838: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_351: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_835, mul_838);  mul_835 = mul_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_839: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_832, 0.5);  mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_352: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_351, mul_839);  add_351 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_1085: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_352, [128, 1024]);  add_352 = None
    permute_998: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1085, [1, 0])
    mm_411: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_998, view_47);  permute_998 = view_47 = None
    permute_999: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_411, [1, 0]);  mm_411 = None
    mm_412: "f32[128, 512]" = torch.ops.aten.mm.default(view_1085, permute_1000);  view_1085 = permute_1000 = None
    view_1086: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_412, [1, 128, 512]);  mm_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_353: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1084, view_1086);  view_1084 = view_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_1001: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_999, [1, 0]);  permute_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_840: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_353, primals_4);  primals_4 = None
    mul_841: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_353, mul_14);  add_353 = mul_14 = None
    sum_127: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 1], True);  mul_841 = None
    view_1087: "f32[512]" = torch.ops.aten.reshape.default(sum_127, [512]);  sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_842: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_840, add_13)
    mul_843: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_840, rsqrt_3);  mul_840 = None
    sum_128: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [2], True);  mul_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_354: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_350, mul_843);  add_350 = mul_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_150: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
    mul_844: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_128, -0.5);  sum_128 = None
    mul_845: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_844, pow_150);  mul_844 = pow_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_134: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_845, [1, 128, 512]);  mul_845 = None
    div_68: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_134, 512);  expand_134 = None
    pow_151: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0);  add_13 = None
    mul_846: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_151, 2.0);  pow_151 = None
    mul_847: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_68, mul_846);  div_68 = mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_355: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_354, mul_847);  add_354 = mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_85: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_848: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
    mul_849: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_355, mul_848);  mul_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_1088: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_849, [128, 512]);  mul_849 = None
    permute_1002: "f32[512, 128]" = torch.ops.aten.permute.default(view_1088, [1, 0])
    mm_413: "f32[512, 384]" = torch.ops.aten.mm.default(permute_1002, view_45);  permute_1002 = view_45 = None
    permute_1003: "f32[384, 512]" = torch.ops.aten.permute.default(mm_413, [1, 0]);  mm_413 = None
    mm_414: "f32[128, 384]" = torch.ops.aten.mm.default(view_1088, permute_1004);  view_1088 = permute_1004 = None
    view_1089: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_414, [1, 128, 384]);  mm_414 = None
    permute_1005: "f32[512, 384]" = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_1090: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1089, [1, 128, 6, 64]);  view_1089 = None
    permute_1006: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1090, [0, 2, 1, 3]);  view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_1091: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_1006, [6, 128, 64]);  permute_1006 = None
    bmm_136: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_1007, view_1091);  permute_1007 = None
    bmm_137: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_1091, permute_1008);  view_1091 = permute_1008 = None
    view_1092: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_136, [1, 6, 128, 64]);  bmm_136 = None
    view_1093: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [1, 6, 128, 128]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_86: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_850: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
    mul_851: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1093, mul_850);  view_1093 = mul_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_852: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_851, alias_160);  mul_851 = None
    sum_129: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [-1], True)
    mul_853: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_160, sum_129);  alias_160 = sum_129 = None
    sub_69: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_24: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_69, 0);  sub_69 = None
    as_strided_scatter_44: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_24, [6, 128, 128], [16384, 128, 1], 0);  squeeze_24 = None
    as_strided_157: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_44, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_44 = None
    new_empty_strided_22: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_157, [6, 128, 128], [16384, 128, 1])
    as_strided_159: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_157, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_45: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_157, as_strided_159, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_356: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_346, as_strided_159);  add_346 = as_strided_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_138: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_1009, as_strided_scatter_45);  permute_1009 = None
    bmm_139: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_45, permute_1010);  as_strided_scatter_45 = permute_1010 = None
    view_1094: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_138, [1, 6, 64, 128]);  bmm_138 = None
    view_1095: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_139, [1, 6, 128, 64]);  bmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_1011: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1094, [0, 1, 3, 2]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1012: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1092, [0, 2, 1, 3]);  view_1092 = None
    clone_187: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
    view_1096: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_187, [1, 128, 384]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1097: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1096, [128, 384]);  view_1096 = None
    permute_1013: "f32[384, 128]" = torch.ops.aten.permute.default(view_1097, [1, 0])
    mm_415: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1013, view_27);  permute_1013 = None
    permute_1014: "f32[512, 384]" = torch.ops.aten.permute.default(mm_415, [1, 0]);  mm_415 = None
    mm_416: "f32[128, 512]" = torch.ops.aten.mm.default(view_1097, permute_1015);  view_1097 = permute_1015 = None
    view_1098: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_416, [1, 128, 512]);  mm_416 = None
    permute_1016: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1014, [1, 0]);  permute_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1017: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_1011, [0, 2, 1, 3]);  permute_1011 = None
    view_1099: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_1017, [1, 128, 384]);  permute_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1100: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1099, [128, 384]);  view_1099 = None
    permute_1018: "f32[384, 128]" = torch.ops.aten.permute.default(view_1100, [1, 0])
    mm_417: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1018, view_27);  permute_1018 = None
    permute_1019: "f32[512, 384]" = torch.ops.aten.permute.default(mm_417, [1, 0]);  mm_417 = None
    mm_418: "f32[128, 512]" = torch.ops.aten.mm.default(view_1100, permute_1020);  view_1100 = permute_1020 = None
    view_1101: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_418, [1, 128, 512]);  mm_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_357: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1098, view_1101);  view_1098 = view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1021: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1019, [1, 0]);  permute_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1022: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1095, [0, 2, 1, 3]);  view_1095 = None
    clone_188: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1022, memory_format = torch.contiguous_format);  permute_1022 = None
    view_1102: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_188, [1, 128, 384]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1103: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1102, [128, 384]);  view_1102 = None
    permute_1023: "f32[384, 128]" = torch.ops.aten.permute.default(view_1103, [1, 0])
    mm_419: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1023, view_27);  permute_1023 = view_27 = None
    permute_1024: "f32[512, 384]" = torch.ops.aten.permute.default(mm_419, [1, 0]);  mm_419 = None
    mm_420: "f32[128, 512]" = torch.ops.aten.mm.default(view_1103, permute_1025);  view_1103 = permute_1025 = None
    view_1104: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_420, [1, 128, 512]);  mm_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_358: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_357, view_1104);  add_357 = view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_1026: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1024, [1, 0]);  permute_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_854: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_358, primals_3);  primals_3 = None
    mul_855: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_358, mul_12);  add_358 = mul_12 = None
    sum_130: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_855, [0, 1], True);  mul_855 = None
    view_1105: "f32[512]" = torch.ops.aten.reshape.default(sum_130, [512]);  sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_856: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_854, add_10)
    mul_857: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_854, rsqrt_2);  mul_854 = None
    sum_131: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_856, [2], True);  mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_359: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_355, mul_857);  add_355 = mul_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_152: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
    mul_858: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_131, -0.5);  sum_131 = None
    mul_859: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_858, pow_152);  mul_858 = pow_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_135: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_859, [1, 128, 512]);  mul_859 = None
    div_69: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_135, 512);  expand_135 = None
    pow_153: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_10, 1.0);  add_10 = None
    mul_860: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_153, 2.0);  pow_153 = None
    mul_861: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_69, mul_860);  div_69 = mul_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_360: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_359, mul_861);  add_359 = mul_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    convert_element_type_87: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
    mul_862: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
    mul_863: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_360, mul_862);  mul_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_1106: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_863, [128, 512]);  mul_863 = None
    permute_1027: "f32[512, 128]" = torch.ops.aten.permute.default(view_1106, [1, 0])
    mm_421: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_1027, view_25);  permute_1027 = view_25 = None
    permute_1028: "f32[1024, 512]" = torch.ops.aten.permute.default(mm_421, [1, 0]);  mm_421 = None
    mm_422: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1106, permute_1029);  view_1106 = permute_1029 = None
    view_1107: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_422, [1, 128, 1024]);  mm_422 = None
    permute_1030: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_88: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_864: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
    mul_865: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1107, mul_864);  view_1107 = mul_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_866: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_865, mul_10);  mul_10 = None
    mul_867: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_865, view_24);  mul_865 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_1108: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_866, [128, 1024]);  mul_866 = None
    permute_1031: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1108, [1, 0])
    mm_423: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_1031, view_21);  permute_1031 = None
    permute_1032: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_423, [1, 0]);  mm_423 = None
    mm_424: "f32[128, 512]" = torch.ops.aten.mm.default(view_1108, permute_1033);  view_1108 = permute_1033 = None
    view_1109: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_424, [1, 128, 512]);  mm_424 = None
    permute_1034: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_1032, [1, 0]);  permute_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_868: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_867, mul_7);  mul_7 = None
    mul_869: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_867, add_9);  mul_867 = add_9 = None
    mul_870: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub_70: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_870);  mul_870 = None
    mul_871: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_868, sub_70);  mul_868 = sub_70 = None
    mul_872: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_871, 0.7978845608028654);  mul_871 = None
    mul_873: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_872, 0.044715)
    pow_154: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 2.0);  view_22 = None
    mul_874: "f32[1, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_154, 3.0);  pow_154 = None
    mul_875: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_873, mul_874);  mul_873 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_361: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_872, mul_875);  mul_872 = mul_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_876: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_869, 0.5);  mul_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_362: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_361, mul_876);  add_361 = mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_1110: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_362, [128, 1024]);  add_362 = None
    permute_1035: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1110, [1, 0])
    mm_425: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_1035, view_21);  permute_1035 = view_21 = None
    permute_1036: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_425, [1, 0]);  mm_425 = None
    mm_426: "f32[128, 512]" = torch.ops.aten.mm.default(view_1110, permute_1037);  view_1110 = permute_1037 = None
    view_1111: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_426, [1, 128, 512]);  mm_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    add_363: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1109, view_1111);  view_1109 = view_1111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_1038: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_1036, [1, 0]);  permute_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_877: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_363, primals_2);  primals_2 = None
    mul_878: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_363, mul_5);  add_363 = mul_5 = None
    sum_132: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1], True);  mul_878 = None
    view_1112: "f32[512]" = torch.ops.aten.reshape.default(sum_132, [512]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_879: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_877, add_6)
    mul_880: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_877, rsqrt_1);  mul_877 = None
    sum_133: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [2], True);  mul_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_364: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_360, mul_880);  add_360 = mul_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_155: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
    mul_881: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_133, -0.5);  sum_133 = None
    mul_882: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_881, pow_155);  mul_881 = pow_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_136: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_882, [1, 128, 512]);  mul_882 = None
    div_70: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_136, 512);  expand_136 = None
    pow_156: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0);  add_6 = None
    mul_883: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_156, 2.0);  pow_156 = None
    mul_884: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_70, mul_883);  div_70 = mul_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_365: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_364, mul_884);  add_364 = mul_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    convert_element_type_89: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_885: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
    mul_886: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_365, mul_885);  mul_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_1113: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_886, [128, 512]);  mul_886 = None
    permute_1039: "f32[512, 128]" = torch.ops.aten.permute.default(view_1113, [1, 0])
    mm_427: "f32[512, 384]" = torch.ops.aten.mm.default(permute_1039, view_19);  permute_1039 = view_19 = None
    permute_1040: "f32[384, 512]" = torch.ops.aten.permute.default(mm_427, [1, 0]);  mm_427 = None
    mm_428: "f32[128, 384]" = torch.ops.aten.mm.default(view_1113, permute_1041);  view_1113 = permute_1041 = None
    view_1114: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_428, [1, 128, 384]);  mm_428 = None
    permute_1042: "f32[512, 384]" = torch.ops.aten.permute.default(permute_1040, [1, 0]);  permute_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    view_1115: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1114, [1, 128, 6, 64]);  view_1114 = None
    permute_1043: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1115, [0, 2, 1, 3]);  view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    view_1116: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(permute_1043, [6, 128, 64]);  permute_1043 = None
    bmm_140: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(permute_1044, view_1116);  permute_1044 = None
    bmm_141: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_1116, permute_1045);  view_1116 = permute_1045 = None
    view_1117: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_140, [1, 6, 128, 64]);  bmm_140 = None
    view_1118: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, [1, 6, 128, 128]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    convert_element_type_90: "f32[1, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_887: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
    mul_888: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1118, mul_887);  view_1118 = mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    mul_889: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_888, alias_164);  mul_888 = None
    sum_134: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [-1], True)
    mul_890: "f32[1, 6, 128, 128]" = torch.ops.aten.mul.Tensor(alias_164, sum_134);  alias_164 = sum_134 = None
    sub_71: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    squeeze_25: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sub_71, 0);  sub_71 = None
    as_strided_scatter_46: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_8, squeeze_25, [6, 128, 128], [16384, 128, 1], 0);  full_8 = squeeze_25 = None
    as_strided_164: "f32[6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_scatter_46, [6, 128, 128], [16384, 128, 1], 0);  as_strided_scatter_46 = None
    new_empty_strided_23: "f32[6, 128, 128]" = torch.ops.aten.new_empty_strided.default(as_strided_164, [6, 128, 128], [16384, 128, 1])
    as_strided_166: "f32[1, 6, 128, 128]" = torch.ops.aten.as_strided.default(as_strided_164, [1, 6, 128, 128], [98304, 16384, 128, 1], 0)
    as_strided_scatter_47: "f32[6, 128, 128]" = torch.ops.aten.as_strided_scatter.default(as_strided_164, as_strided_166, [1, 6, 128, 128], [98304, 16384, 128, 1], 0);  as_strided_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_366: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_356, as_strided_166);  add_356 = as_strided_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    squeeze_26: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(add_366, 0);  add_366 = None
    permute_1046: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_26, [1, 2, 0]);  squeeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    where_8: "f32[128, 128, 6]" = torch.ops.aten.where.self(full_default_11, full_default_7, permute_1046);  full_default_11 = permute_1046 = None
    clone_194: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_8, memory_format = torch.contiguous_format);  where_8 = None
    _unsafe_index_put_2: "f32[32, 6]" = torch.ops.prims._unsafe_index_put_.default(full_default_13, [add_3], clone_194, True);  full_default_13 = add_3 = clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    bmm_142: "f32[6, 64, 128]" = torch.ops.aten.bmm.default(permute_1047, as_strided_scatter_47);  permute_1047 = None
    bmm_143: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(as_strided_scatter_47, permute_1048);  as_strided_scatter_47 = permute_1048 = None
    view_1119: "f32[1, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_142, [1, 6, 64, 128]);  bmm_142 = None
    view_1120: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_143, [1, 6, 128, 64]);  bmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_1049: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_1119, [0, 1, 3, 2]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1050: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1117, [0, 2, 1, 3]);  view_1117 = None
    clone_195: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1050, memory_format = torch.contiguous_format);  permute_1050 = None
    view_1121: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_195, [1, 128, 384]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1122: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1121, [128, 384]);  view_1121 = None
    permute_1051: "f32[384, 128]" = torch.ops.aten.permute.default(view_1122, [1, 0])
    mm_429: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1051, view_1);  permute_1051 = None
    permute_1052: "f32[512, 384]" = torch.ops.aten.permute.default(mm_429, [1, 0]);  mm_429 = None
    mm_430: "f32[128, 512]" = torch.ops.aten.mm.default(view_1122, permute_1053);  view_1122 = permute_1053 = None
    view_1123: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_430, [1, 128, 512]);  mm_430 = None
    permute_1054: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1052, [1, 0]);  permute_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1055: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(permute_1049, [0, 2, 1, 3]);  permute_1049 = None
    view_1124: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(permute_1055, [1, 128, 384]);  permute_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_1125: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1124, [128, 384]);  view_1124 = None
    permute_1056: "f32[384, 128]" = torch.ops.aten.permute.default(view_1125, [1, 0])
    mm_431: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1056, view_1);  permute_1056 = None
    permute_1057: "f32[512, 384]" = torch.ops.aten.permute.default(mm_431, [1, 0]);  mm_431 = None
    mm_432: "f32[128, 512]" = torch.ops.aten.mm.default(view_1125, permute_1058);  view_1125 = permute_1058 = None
    view_1126: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_432, [1, 128, 512]);  mm_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    add_367: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1123, view_1126);  view_1123 = view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1059: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1057, [1, 0]);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    permute_1060: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_1120, [0, 2, 1, 3]);  view_1120 = None
    clone_196: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1060, memory_format = torch.contiguous_format);  permute_1060 = None
    view_1127: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_196, [1, 128, 384]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1128: "f32[128, 384]" = torch.ops.aten.reshape.default(view_1127, [128, 384]);  view_1127 = None
    permute_1061: "f32[384, 128]" = torch.ops.aten.permute.default(view_1128, [1, 0])
    mm_433: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1061, view_1);  permute_1061 = view_1 = None
    permute_1062: "f32[512, 384]" = torch.ops.aten.permute.default(mm_433, [1, 0]);  mm_433 = None
    mm_434: "f32[128, 512]" = torch.ops.aten.mm.default(view_1128, permute_1063);  view_1128 = permute_1063 = None
    view_1129: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_434, [1, 128, 512]);  mm_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    add_368: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_367, view_1129);  add_367 = view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_1064: "f32[384, 512]" = torch.ops.aten.permute.default(permute_1062, [1, 0]);  permute_1062 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_891: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_368, primals_1);  primals_1 = None
    mul_892: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_368, mul_1);  add_368 = mul_1 = None
    sum_135: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1], True);  mul_892 = None
    view_1130: "f32[512]" = torch.ops.aten.reshape.default(sum_135, [512]);  sum_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    mul_893: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_891, getitem)
    mul_894: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_891, rsqrt);  mul_891 = None
    sum_136: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_893, [2], True);  mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_369: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_365, mul_894);  add_365 = mul_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    pow_157: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
    mul_895: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_136, -0.5);  sum_136 = None
    mul_896: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_895, pow_157);  mul_895 = pow_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expand_137: "f32[1, 128, 512]" = torch.ops.aten.expand.default(mul_896, [1, 128, 512]);  mul_896 = None
    div_71: "f32[1, 128, 512]" = torch.ops.aten.div.Scalar(expand_137, 512);  expand_137 = None
    pow_158: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem, 1.0);  getitem = None
    mul_897: "f32[1, 128, 512]" = torch.ops.aten.mul.Scalar(pow_158, 2.0);  pow_158 = None
    mul_898: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(div_71, mul_897);  div_71 = mul_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    add_370: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_369, mul_898);  add_369 = mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    convert_element_type_91: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_899: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
    mul_900: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_370, mul_899);  add_370 = mul_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    eq_3: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_22: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
    where_9: "f32[1, 128, 512]" = torch.ops.aten.where.self(unsqueeze_22, full_default_7, mul_900);  unsqueeze_22 = full_default_7 = mul_900 = None
    _unsafe_index_put_3: "f32[250112, 512]" = torch.ops.prims._unsafe_index_put_.default(full_default_15, [view], where_9, True);  full_default_15 = view = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    add_371: "f32[250112, 512]" = torch.ops.aten.add.Tensor(_unsafe_index_put_1, _unsafe_index_put_3);  _unsafe_index_put_1 = _unsafe_index_put_3 = None
    return [view_1130, view_1112, view_1105, view_1087, view_1080, view_1062, view_1055, view_1037, view_1030, view_1012, view_1005, view_987, view_980, view_962, view_955, view_937, view_930, view_929, view_911, view_893, view_886, view_868, view_850, view_843, view_825, view_807, view_800, view_782, view_764, view_757, view_739, view_721, view_714, view_696, view_678, view_671, view_653, view_635, view_628, view_610, view_592, view_585, add_371, permute_1064, permute_1059, permute_1054, _unsafe_index_put_2, permute_1042, permute_1038, permute_1034, permute_1030, permute_1026, permute_1021, permute_1016, permute_1005, permute_1001, permute_997, permute_993, permute_989, permute_984, permute_979, permute_968, permute_964, permute_960, permute_956, permute_952, permute_947, permute_942, permute_931, permute_927, permute_923, permute_919, permute_915, permute_910, permute_905, permute_894, permute_890, permute_886, permute_882, permute_878, permute_873, permute_868, permute_857, permute_853, permute_849, permute_845, permute_841, permute_836, permute_831, permute_820, permute_816, permute_812, permute_808, permute_804, permute_799, permute_794, permute_783, permute_779, permute_775, permute_771, permute_767, permute_762, permute_757, _unsafe_index_put, permute_745, permute_741, permute_736, permute_731, permute_720, permute_716, permute_712, permute_708, permute_704, permute_699, permute_694, permute_683, permute_679, permute_674, permute_669, permute_658, permute_654, permute_650, permute_646, permute_642, permute_637, permute_632, permute_621, permute_617, permute_612, permute_607, permute_596, permute_592, permute_588, permute_584, permute_580, permute_575, permute_570, permute_559, permute_555, permute_550, permute_545, permute_534, permute_530, permute_526, permute_522, permute_518, permute_513, permute_508, permute_497, permute_493, permute_488, permute_483, permute_472, permute_468, permute_464, permute_460, permute_456, permute_451, permute_446, permute_435, permute_431, permute_426, permute_421, permute_410, permute_406, permute_402, permute_398, permute_394, permute_389, permute_384, permute_373, permute_369, permute_364, permute_359, permute_348, permute_344, permute_340, permute_336, permute_332, permute_327, permute_322, permute_311, permute_307, permute_302, permute_297, permute_286, permute_282, permute_278, permute_274, permute_270, None, None, None]
    