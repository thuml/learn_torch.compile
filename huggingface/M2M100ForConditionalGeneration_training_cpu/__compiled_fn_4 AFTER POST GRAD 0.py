from __future__ import annotations



def forward(self, primals_2: "f32[1024]", primals_12: "f32[1024]", primals_18: "f32[1024]", primals_28: "f32[1024]", primals_34: "f32[1024]", primals_44: "f32[1024]", primals_50: "f32[1024]", primals_60: "f32[1024]", primals_66: "f32[1024]", primals_76: "f32[1024]", primals_82: "f32[1024]", primals_92: "f32[1024]", primals_98: "f32[1024]", primals_108: "f32[1024]", primals_114: "f32[1024]", primals_124: "f32[1024]", primals_130: "f32[1024]", primals_140: "f32[1024]", primals_146: "f32[1024]", primals_156: "f32[1024]", primals_162: "f32[1024]", primals_172: "f32[1024]", primals_178: "f32[1024]", primals_188: "f32[1024]", primals_194: "f32[1024]", primals_197: "f32[1024]", primals_207: "f32[1024]", primals_217: "f32[1024]", primals_223: "f32[1024]", primals_233: "f32[1024]", primals_243: "f32[1024]", primals_249: "f32[1024]", primals_259: "f32[1024]", primals_269: "f32[1024]", primals_275: "f32[1024]", primals_285: "f32[1024]", primals_295: "f32[1024]", primals_301: "f32[1024]", primals_311: "f32[1024]", primals_321: "f32[1024]", primals_327: "f32[1024]", primals_337: "f32[1024]", primals_347: "f32[1024]", primals_353: "f32[1024]", primals_363: "f32[1024]", primals_373: "f32[1024]", primals_379: "f32[1024]", primals_389: "f32[1024]", primals_399: "f32[1024]", primals_405: "f32[1024]", primals_415: "f32[1024]", primals_425: "f32[1024]", primals_431: "f32[1024]", primals_441: "f32[1024]", primals_451: "f32[1024]", primals_457: "f32[1024]", primals_467: "f32[1024]", primals_477: "f32[1024]", primals_483: "f32[1024]", primals_493: "f32[1024]", primals_503: "f32[1024]", primals_509: "f32[1024]", primals_514: "i64[1, 128]", view: "i64[1, 128]", mul_2: "f32[1, 128, 1024]", view_3: "f32[128, 1024]", bmm: "f32[16, 128, 128]", amax: "f32[16, 128, 1]", sum_1: "f32[16, 128, 1]", view_17: "f32[128, 1024]", mul_5: "f32[1, 128, 1024]", view_19: "f32[128, 1024]", view_21: "f32[128, 4096]", mul_7: "f32[1, 128, 1024]", view_23: "f32[128, 1024]", bmm_2: "f32[16, 128, 128]", amax_1: "f32[16, 128, 1]", sum_2: "f32[16, 128, 1]", view_37: "f32[128, 1024]", mul_10: "f32[1, 128, 1024]", view_39: "f32[128, 1024]", view_41: "f32[128, 4096]", mul_12: "f32[1, 128, 1024]", view_43: "f32[128, 1024]", bmm_4: "f32[16, 128, 128]", amax_2: "f32[16, 128, 1]", sum_3: "f32[16, 128, 1]", view_57: "f32[128, 1024]", mul_15: "f32[1, 128, 1024]", view_59: "f32[128, 1024]", view_61: "f32[128, 4096]", mul_17: "f32[1, 128, 1024]", view_63: "f32[128, 1024]", bmm_6: "f32[16, 128, 128]", amax_3: "f32[16, 128, 1]", sum_4: "f32[16, 128, 1]", view_77: "f32[128, 1024]", mul_20: "f32[1, 128, 1024]", view_79: "f32[128, 1024]", view_81: "f32[128, 4096]", mul_22: "f32[1, 128, 1024]", view_83: "f32[128, 1024]", bmm_8: "f32[16, 128, 128]", amax_4: "f32[16, 128, 1]", sum_5: "f32[16, 128, 1]", view_97: "f32[128, 1024]", mul_25: "f32[1, 128, 1024]", view_99: "f32[128, 1024]", view_101: "f32[128, 4096]", mul_27: "f32[1, 128, 1024]", view_103: "f32[128, 1024]", bmm_10: "f32[16, 128, 128]", amax_5: "f32[16, 128, 1]", sum_6: "f32[16, 128, 1]", view_117: "f32[128, 1024]", mul_30: "f32[1, 128, 1024]", view_119: "f32[128, 1024]", view_121: "f32[128, 4096]", mul_32: "f32[1, 128, 1024]", view_123: "f32[128, 1024]", bmm_12: "f32[16, 128, 128]", amax_6: "f32[16, 128, 1]", sum_7: "f32[16, 128, 1]", view_137: "f32[128, 1024]", mul_35: "f32[1, 128, 1024]", view_139: "f32[128, 1024]", view_141: "f32[128, 4096]", mul_37: "f32[1, 128, 1024]", view_143: "f32[128, 1024]", bmm_14: "f32[16, 128, 128]", amax_7: "f32[16, 128, 1]", sum_8: "f32[16, 128, 1]", view_157: "f32[128, 1024]", mul_40: "f32[1, 128, 1024]", view_159: "f32[128, 1024]", view_161: "f32[128, 4096]", mul_42: "f32[1, 128, 1024]", view_163: "f32[128, 1024]", bmm_16: "f32[16, 128, 128]", amax_8: "f32[16, 128, 1]", sum_9: "f32[16, 128, 1]", view_177: "f32[128, 1024]", mul_45: "f32[1, 128, 1024]", view_179: "f32[128, 1024]", view_181: "f32[128, 4096]", mul_47: "f32[1, 128, 1024]", view_183: "f32[128, 1024]", bmm_18: "f32[16, 128, 128]", amax_9: "f32[16, 128, 1]", sum_10: "f32[16, 128, 1]", view_197: "f32[128, 1024]", mul_50: "f32[1, 128, 1024]", view_199: "f32[128, 1024]", view_201: "f32[128, 4096]", mul_52: "f32[1, 128, 1024]", view_203: "f32[128, 1024]", bmm_20: "f32[16, 128, 128]", amax_10: "f32[16, 128, 1]", sum_11: "f32[16, 128, 1]", view_217: "f32[128, 1024]", mul_55: "f32[1, 128, 1024]", view_219: "f32[128, 1024]", view_221: "f32[128, 4096]", mul_57: "f32[1, 128, 1024]", view_223: "f32[128, 1024]", bmm_22: "f32[16, 128, 128]", amax_11: "f32[16, 128, 1]", sum_12: "f32[16, 128, 1]", view_237: "f32[128, 1024]", mul_60: "f32[1, 128, 1024]", view_239: "f32[128, 1024]", view_241: "f32[128, 4096]", mul_62: "f32[1, 128, 1024]", view_243: "i64[1, 128]", mul_66: "f32[1, 128, 1024]", view_247: "f32[128, 1024]", view_263: "f32[128, 1024]", mul_69: "f32[1, 128, 1024]", view_265: "f32[128, 1024]", view_267: "f32[128, 1024]", bmm_26: "f32[16, 128, 128]", amax_13: "f32[16, 128, 1]", sum_14: "f32[16, 128, 1]", view_279: "f32[128, 1024]", mul_72: "f32[1, 128, 1024]", view_281: "f32[128, 1024]", view_283: "f32[128, 4096]", mul_74: "f32[1, 128, 1024]", view_285: "f32[128, 1024]", view_301: "f32[128, 1024]", mul_77: "f32[1, 128, 1024]", view_303: "f32[128, 1024]", bmm_30: "f32[16, 128, 128]", amax_15: "f32[16, 128, 1]", sum_16: "f32[16, 128, 1]", view_317: "f32[128, 1024]", mul_80: "f32[1, 128, 1024]", view_319: "f32[128, 1024]", view_321: "f32[128, 4096]", mul_82: "f32[1, 128, 1024]", view_323: "f32[128, 1024]", view_339: "f32[128, 1024]", mul_85: "f32[1, 128, 1024]", view_341: "f32[128, 1024]", bmm_34: "f32[16, 128, 128]", amax_17: "f32[16, 128, 1]", sum_18: "f32[16, 128, 1]", view_355: "f32[128, 1024]", mul_88: "f32[1, 128, 1024]", view_357: "f32[128, 1024]", view_359: "f32[128, 4096]", mul_90: "f32[1, 128, 1024]", view_361: "f32[128, 1024]", view_377: "f32[128, 1024]", mul_93: "f32[1, 128, 1024]", view_379: "f32[128, 1024]", bmm_38: "f32[16, 128, 128]", amax_19: "f32[16, 128, 1]", sum_20: "f32[16, 128, 1]", view_393: "f32[128, 1024]", mul_96: "f32[1, 128, 1024]", view_395: "f32[128, 1024]", view_397: "f32[128, 4096]", mul_98: "f32[1, 128, 1024]", view_399: "f32[128, 1024]", view_415: "f32[128, 1024]", mul_101: "f32[1, 128, 1024]", view_417: "f32[128, 1024]", bmm_42: "f32[16, 128, 128]", amax_21: "f32[16, 128, 1]", sum_22: "f32[16, 128, 1]", view_431: "f32[128, 1024]", mul_104: "f32[1, 128, 1024]", view_433: "f32[128, 1024]", view_435: "f32[128, 4096]", mul_106: "f32[1, 128, 1024]", view_437: "f32[128, 1024]", view_453: "f32[128, 1024]", mul_109: "f32[1, 128, 1024]", view_455: "f32[128, 1024]", bmm_46: "f32[16, 128, 128]", amax_23: "f32[16, 128, 1]", sum_24: "f32[16, 128, 1]", view_469: "f32[128, 1024]", mul_112: "f32[1, 128, 1024]", view_471: "f32[128, 1024]", view_473: "f32[128, 4096]", mul_114: "f32[1, 128, 1024]", view_475: "f32[128, 1024]", view_491: "f32[128, 1024]", mul_117: "f32[1, 128, 1024]", view_493: "f32[128, 1024]", bmm_50: "f32[16, 128, 128]", amax_25: "f32[16, 128, 1]", sum_26: "f32[16, 128, 1]", view_507: "f32[128, 1024]", mul_120: "f32[1, 128, 1024]", view_509: "f32[128, 1024]", view_511: "f32[128, 4096]", mul_122: "f32[1, 128, 1024]", view_513: "f32[128, 1024]", view_529: "f32[128, 1024]", mul_125: "f32[1, 128, 1024]", view_531: "f32[128, 1024]", bmm_54: "f32[16, 128, 128]", amax_27: "f32[16, 128, 1]", sum_28: "f32[16, 128, 1]", view_545: "f32[128, 1024]", mul_128: "f32[1, 128, 1024]", view_547: "f32[128, 1024]", view_549: "f32[128, 4096]", mul_130: "f32[1, 128, 1024]", view_551: "f32[128, 1024]", view_567: "f32[128, 1024]", mul_133: "f32[1, 128, 1024]", view_569: "f32[128, 1024]", bmm_58: "f32[16, 128, 128]", amax_29: "f32[16, 128, 1]", sum_30: "f32[16, 128, 1]", view_583: "f32[128, 1024]", mul_136: "f32[1, 128, 1024]", view_585: "f32[128, 1024]", view_587: "f32[128, 4096]", mul_138: "f32[1, 128, 1024]", view_589: "f32[128, 1024]", view_605: "f32[128, 1024]", mul_141: "f32[1, 128, 1024]", view_607: "f32[128, 1024]", bmm_62: "f32[16, 128, 128]", amax_31: "f32[16, 128, 1]", sum_32: "f32[16, 128, 1]", view_621: "f32[128, 1024]", mul_144: "f32[1, 128, 1024]", view_623: "f32[128, 1024]", view_625: "f32[128, 4096]", mul_146: "f32[1, 128, 1024]", view_627: "f32[128, 1024]", view_643: "f32[128, 1024]", mul_149: "f32[1, 128, 1024]", view_645: "f32[128, 1024]", bmm_66: "f32[16, 128, 128]", amax_33: "f32[16, 128, 1]", sum_34: "f32[16, 128, 1]", view_659: "f32[128, 1024]", mul_152: "f32[1, 128, 1024]", view_661: "f32[128, 1024]", view_663: "f32[128, 4096]", mul_154: "f32[1, 128, 1024]", view_665: "f32[128, 1024]", view_681: "f32[128, 1024]", mul_157: "f32[1, 128, 1024]", view_683: "f32[128, 1024]", bmm_70: "f32[16, 128, 128]", amax_35: "f32[16, 128, 1]", sum_36: "f32[16, 128, 1]", view_697: "f32[128, 1024]", mul_160: "f32[1, 128, 1024]", view_699: "f32[128, 1024]", view_701: "f32[128, 4096]", mul_162: "f32[1, 128, 1024]", view_703: "f32[128, 1024]", sub_99: "f32[128, 128112]", convert_element_type_6: "f32[]", permute_375: "f32[128112, 1024]", div_38: "f32[1, 128, 1]", permute_377: "f32[1024, 4096]", le: "b8[1, 128, 4096]", permute_381: "f32[4096, 1024]", div_39: "f32[1, 128, 1]", permute_385: "f32[1024, 1024]", permute_390: "f32[16, 128, 128]", permute_391: "f32[16, 64, 128]", permute_392: "f32[16, 64, 128]", permute_393: "f32[16, 128, 64]", permute_397: "f32[1024, 1024]", permute_402: "f32[1024, 1024]", permute_406: "f32[1024, 1024]", div_40: "f32[1, 128, 1]", permute_410: "f32[1024, 1024]", permute_415: "f32[16, 128, 128]", permute_416: "f32[16, 64, 128]", alias_68: "f32[16, 128, 128]", permute_417: "f32[16, 64, 128]", permute_418: "f32[16, 128, 64]", permute_422: "f32[1024, 1024]", permute_427: "f32[1024, 1024]", permute_431: "f32[1024, 1024]", div_41: "f32[1, 128, 1]", permute_435: "f32[1024, 4096]", le_1: "b8[1, 128, 4096]", permute_439: "f32[4096, 1024]", div_42: "f32[1, 128, 1]", permute_443: "f32[1024, 1024]", permute_448: "f32[16, 128, 128]", permute_449: "f32[16, 64, 128]", permute_450: "f32[16, 64, 128]", permute_451: "f32[16, 128, 64]", permute_455: "f32[1024, 1024]", permute_460: "f32[1024, 1024]", permute_464: "f32[1024, 1024]", div_43: "f32[1, 128, 1]", permute_468: "f32[1024, 1024]", permute_473: "f32[16, 128, 128]", permute_474: "f32[16, 64, 128]", alias_71: "f32[16, 128, 128]", permute_475: "f32[16, 64, 128]", permute_476: "f32[16, 128, 64]", permute_480: "f32[1024, 1024]", permute_485: "f32[1024, 1024]", permute_489: "f32[1024, 1024]", div_44: "f32[1, 128, 1]", permute_493: "f32[1024, 4096]", le_2: "b8[1, 128, 4096]", permute_497: "f32[4096, 1024]", div_45: "f32[1, 128, 1]", permute_501: "f32[1024, 1024]", permute_506: "f32[16, 128, 128]", permute_507: "f32[16, 64, 128]", permute_508: "f32[16, 64, 128]", permute_509: "f32[16, 128, 64]", permute_513: "f32[1024, 1024]", permute_518: "f32[1024, 1024]", permute_522: "f32[1024, 1024]", div_46: "f32[1, 128, 1]", permute_526: "f32[1024, 1024]", permute_531: "f32[16, 128, 128]", permute_532: "f32[16, 64, 128]", alias_74: "f32[16, 128, 128]", permute_533: "f32[16, 64, 128]", permute_534: "f32[16, 128, 64]", permute_538: "f32[1024, 1024]", permute_543: "f32[1024, 1024]", permute_547: "f32[1024, 1024]", div_47: "f32[1, 128, 1]", permute_551: "f32[1024, 4096]", le_3: "b8[1, 128, 4096]", permute_555: "f32[4096, 1024]", div_48: "f32[1, 128, 1]", permute_559: "f32[1024, 1024]", permute_564: "f32[16, 128, 128]", permute_565: "f32[16, 64, 128]", permute_566: "f32[16, 64, 128]", permute_567: "f32[16, 128, 64]", permute_571: "f32[1024, 1024]", permute_576: "f32[1024, 1024]", permute_580: "f32[1024, 1024]", div_49: "f32[1, 128, 1]", permute_584: "f32[1024, 1024]", permute_589: "f32[16, 128, 128]", permute_590: "f32[16, 64, 128]", alias_77: "f32[16, 128, 128]", permute_591: "f32[16, 64, 128]", permute_592: "f32[16, 128, 64]", permute_596: "f32[1024, 1024]", permute_601: "f32[1024, 1024]", permute_605: "f32[1024, 1024]", div_50: "f32[1, 128, 1]", permute_609: "f32[1024, 4096]", le_4: "b8[1, 128, 4096]", permute_613: "f32[4096, 1024]", div_51: "f32[1, 128, 1]", permute_617: "f32[1024, 1024]", permute_622: "f32[16, 128, 128]", permute_623: "f32[16, 64, 128]", permute_624: "f32[16, 64, 128]", permute_625: "f32[16, 128, 64]", permute_629: "f32[1024, 1024]", permute_634: "f32[1024, 1024]", permute_638: "f32[1024, 1024]", div_52: "f32[1, 128, 1]", permute_642: "f32[1024, 1024]", permute_647: "f32[16, 128, 128]", permute_648: "f32[16, 64, 128]", alias_80: "f32[16, 128, 128]", permute_649: "f32[16, 64, 128]", permute_650: "f32[16, 128, 64]", permute_654: "f32[1024, 1024]", permute_659: "f32[1024, 1024]", permute_663: "f32[1024, 1024]", div_53: "f32[1, 128, 1]", permute_667: "f32[1024, 4096]", le_5: "b8[1, 128, 4096]", permute_671: "f32[4096, 1024]", div_54: "f32[1, 128, 1]", permute_675: "f32[1024, 1024]", permute_680: "f32[16, 128, 128]", permute_681: "f32[16, 64, 128]", permute_682: "f32[16, 64, 128]", permute_683: "f32[16, 128, 64]", permute_687: "f32[1024, 1024]", permute_692: "f32[1024, 1024]", permute_696: "f32[1024, 1024]", div_55: "f32[1, 128, 1]", permute_700: "f32[1024, 1024]", permute_705: "f32[16, 128, 128]", permute_706: "f32[16, 64, 128]", alias_83: "f32[16, 128, 128]", permute_707: "f32[16, 64, 128]", permute_708: "f32[16, 128, 64]", permute_712: "f32[1024, 1024]", permute_717: "f32[1024, 1024]", permute_721: "f32[1024, 1024]", div_56: "f32[1, 128, 1]", permute_725: "f32[1024, 4096]", le_6: "b8[1, 128, 4096]", permute_729: "f32[4096, 1024]", div_57: "f32[1, 128, 1]", permute_733: "f32[1024, 1024]", permute_738: "f32[16, 128, 128]", permute_739: "f32[16, 64, 128]", permute_740: "f32[16, 64, 128]", permute_741: "f32[16, 128, 64]", permute_745: "f32[1024, 1024]", permute_750: "f32[1024, 1024]", permute_754: "f32[1024, 1024]", div_58: "f32[1, 128, 1]", permute_758: "f32[1024, 1024]", permute_763: "f32[16, 128, 128]", permute_764: "f32[16, 64, 128]", alias_86: "f32[16, 128, 128]", permute_765: "f32[16, 64, 128]", permute_766: "f32[16, 128, 64]", permute_770: "f32[1024, 1024]", permute_775: "f32[1024, 1024]", permute_779: "f32[1024, 1024]", div_59: "f32[1, 128, 1]", permute_783: "f32[1024, 4096]", le_7: "b8[1, 128, 4096]", permute_787: "f32[4096, 1024]", div_60: "f32[1, 128, 1]", permute_791: "f32[1024, 1024]", permute_796: "f32[16, 128, 128]", permute_797: "f32[16, 64, 128]", permute_798: "f32[16, 64, 128]", permute_799: "f32[16, 128, 64]", permute_803: "f32[1024, 1024]", permute_808: "f32[1024, 1024]", permute_812: "f32[1024, 1024]", div_61: "f32[1, 128, 1]", permute_816: "f32[1024, 1024]", permute_821: "f32[16, 128, 128]", permute_822: "f32[16, 64, 128]", alias_89: "f32[16, 128, 128]", permute_823: "f32[16, 64, 128]", permute_824: "f32[16, 128, 64]", permute_828: "f32[1024, 1024]", permute_833: "f32[1024, 1024]", permute_837: "f32[1024, 1024]", div_62: "f32[1, 128, 1]", permute_841: "f32[1024, 4096]", le_8: "b8[1, 128, 4096]", permute_845: "f32[4096, 1024]", div_63: "f32[1, 128, 1]", permute_849: "f32[1024, 1024]", permute_854: "f32[16, 128, 128]", permute_855: "f32[16, 64, 128]", permute_856: "f32[16, 64, 128]", permute_857: "f32[16, 128, 64]", permute_861: "f32[1024, 1024]", permute_866: "f32[1024, 1024]", permute_870: "f32[1024, 1024]", div_64: "f32[1, 128, 1]", permute_874: "f32[1024, 1024]", permute_879: "f32[16, 128, 128]", permute_880: "f32[16, 64, 128]", alias_92: "f32[16, 128, 128]", permute_881: "f32[16, 64, 128]", permute_882: "f32[16, 128, 64]", permute_886: "f32[1024, 1024]", permute_891: "f32[1024, 1024]", permute_895: "f32[1024, 1024]", div_65: "f32[1, 128, 1]", permute_899: "f32[1024, 4096]", le_9: "b8[1, 128, 4096]", permute_903: "f32[4096, 1024]", div_66: "f32[1, 128, 1]", permute_907: "f32[1024, 1024]", permute_912: "f32[16, 128, 128]", permute_913: "f32[16, 64, 128]", permute_914: "f32[16, 64, 128]", permute_915: "f32[16, 128, 64]", permute_919: "f32[1024, 1024]", permute_924: "f32[1024, 1024]", permute_928: "f32[1024, 1024]", div_67: "f32[1, 128, 1]", permute_932: "f32[1024, 1024]", permute_937: "f32[16, 128, 128]", permute_938: "f32[16, 64, 128]", alias_95: "f32[16, 128, 128]", permute_939: "f32[16, 64, 128]", permute_940: "f32[16, 128, 64]", permute_944: "f32[1024, 1024]", permute_949: "f32[1024, 1024]", permute_953: "f32[1024, 1024]", div_68: "f32[1, 128, 1]", permute_957: "f32[1024, 4096]", le_10: "b8[1, 128, 4096]", permute_961: "f32[4096, 1024]", div_69: "f32[1, 128, 1]", permute_965: "f32[1024, 1024]", permute_970: "f32[16, 128, 128]", permute_971: "f32[16, 64, 128]", permute_972: "f32[16, 64, 128]", permute_973: "f32[16, 128, 64]", permute_977: "f32[1024, 1024]", permute_982: "f32[1024, 1024]", permute_986: "f32[1024, 1024]", div_70: "f32[1, 128, 1]", permute_990: "f32[1024, 1024]", permute_995: "f32[16, 128, 128]", permute_996: "f32[16, 64, 128]", alias_98: "f32[16, 128, 128]", permute_997: "f32[16, 64, 128]", permute_998: "f32[16, 128, 64]", permute_1002: "f32[1024, 1024]", permute_1007: "f32[1024, 1024]", permute_1011: "f32[1024, 1024]", div_71: "f32[1, 128, 1]", permute_1015: "f32[1024, 4096]", le_11: "b8[1, 128, 4096]", permute_1019: "f32[4096, 1024]", div_72: "f32[1, 128, 1]", permute_1023: "f32[1024, 1024]", permute_1028: "f32[16, 128, 128]", permute_1029: "f32[16, 64, 128]", permute_1030: "f32[16, 64, 128]", permute_1031: "f32[16, 128, 64]", permute_1035: "f32[1024, 1024]", permute_1040: "f32[1024, 1024]", permute_1044: "f32[1024, 1024]", div_73: "f32[1, 128, 1]", permute_1048: "f32[1024, 1024]", permute_1053: "f32[16, 128, 128]", permute_1054: "f32[16, 64, 128]", alias_101: "f32[16, 128, 128]", permute_1055: "f32[16, 64, 128]", permute_1056: "f32[16, 128, 64]", permute_1060: "f32[1024, 1024]", permute_1065: "f32[1024, 1024]", permute_1069: "f32[1024, 1024]", div_74: "f32[1, 128, 1]", div_75: "f32[1, 128, 1]", permute_1073: "f32[1024, 4096]", le_12: "b8[1, 128, 4096]", permute_1077: "f32[4096, 1024]", div_76: "f32[1, 128, 1]", permute_1081: "f32[1024, 1024]", permute_1086: "f32[16, 128, 128]", permute_1087: "f32[16, 64, 128]", permute_1088: "f32[16, 64, 128]", permute_1089: "f32[16, 128, 64]", permute_1093: "f32[1024, 1024]", permute_1098: "f32[1024, 1024]", permute_1102: "f32[1024, 1024]", div_77: "f32[1, 128, 1]", permute_1106: "f32[1024, 4096]", le_13: "b8[1, 128, 4096]", permute_1110: "f32[4096, 1024]", div_78: "f32[1, 128, 1]", permute_1114: "f32[1024, 1024]", permute_1119: "f32[16, 128, 128]", permute_1120: "f32[16, 64, 128]", permute_1121: "f32[16, 64, 128]", permute_1122: "f32[16, 128, 64]", permute_1126: "f32[1024, 1024]", permute_1131: "f32[1024, 1024]", permute_1135: "f32[1024, 1024]", div_79: "f32[1, 128, 1]", permute_1139: "f32[1024, 4096]", le_14: "b8[1, 128, 4096]", permute_1143: "f32[4096, 1024]", div_80: "f32[1, 128, 1]", permute_1147: "f32[1024, 1024]", permute_1152: "f32[16, 128, 128]", permute_1153: "f32[16, 64, 128]", permute_1154: "f32[16, 64, 128]", permute_1155: "f32[16, 128, 64]", permute_1159: "f32[1024, 1024]", permute_1164: "f32[1024, 1024]", permute_1168: "f32[1024, 1024]", div_81: "f32[1, 128, 1]", permute_1172: "f32[1024, 4096]", le_15: "b8[1, 128, 4096]", permute_1176: "f32[4096, 1024]", div_82: "f32[1, 128, 1]", permute_1180: "f32[1024, 1024]", permute_1185: "f32[16, 128, 128]", permute_1186: "f32[16, 64, 128]", permute_1187: "f32[16, 64, 128]", permute_1188: "f32[16, 128, 64]", permute_1192: "f32[1024, 1024]", permute_1197: "f32[1024, 1024]", permute_1201: "f32[1024, 1024]", div_83: "f32[1, 128, 1]", permute_1205: "f32[1024, 4096]", le_16: "b8[1, 128, 4096]", permute_1209: "f32[4096, 1024]", div_84: "f32[1, 128, 1]", permute_1213: "f32[1024, 1024]", permute_1218: "f32[16, 128, 128]", permute_1219: "f32[16, 64, 128]", permute_1220: "f32[16, 64, 128]", permute_1221: "f32[16, 128, 64]", permute_1225: "f32[1024, 1024]", permute_1230: "f32[1024, 1024]", permute_1234: "f32[1024, 1024]", div_85: "f32[1, 128, 1]", permute_1238: "f32[1024, 4096]", le_17: "b8[1, 128, 4096]", permute_1242: "f32[4096, 1024]", div_86: "f32[1, 128, 1]", permute_1246: "f32[1024, 1024]", permute_1251: "f32[16, 128, 128]", permute_1252: "f32[16, 64, 128]", permute_1253: "f32[16, 64, 128]", permute_1254: "f32[16, 128, 64]", permute_1258: "f32[1024, 1024]", permute_1263: "f32[1024, 1024]", permute_1267: "f32[1024, 1024]", div_87: "f32[1, 128, 1]", permute_1271: "f32[1024, 4096]", le_18: "b8[1, 128, 4096]", permute_1275: "f32[4096, 1024]", div_88: "f32[1, 128, 1]", permute_1279: "f32[1024, 1024]", permute_1284: "f32[16, 128, 128]", permute_1285: "f32[16, 64, 128]", permute_1286: "f32[16, 64, 128]", permute_1287: "f32[16, 128, 64]", permute_1291: "f32[1024, 1024]", permute_1296: "f32[1024, 1024]", permute_1300: "f32[1024, 1024]", div_89: "f32[1, 128, 1]", permute_1304: "f32[1024, 4096]", le_19: "b8[1, 128, 4096]", permute_1308: "f32[4096, 1024]", div_90: "f32[1, 128, 1]", permute_1312: "f32[1024, 1024]", permute_1317: "f32[16, 128, 128]", permute_1318: "f32[16, 64, 128]", permute_1319: "f32[16, 64, 128]", permute_1320: "f32[16, 128, 64]", permute_1324: "f32[1024, 1024]", permute_1329: "f32[1024, 1024]", permute_1333: "f32[1024, 1024]", div_91: "f32[1, 128, 1]", permute_1337: "f32[1024, 4096]", le_20: "b8[1, 128, 4096]", permute_1341: "f32[4096, 1024]", div_92: "f32[1, 128, 1]", permute_1345: "f32[1024, 1024]", permute_1350: "f32[16, 128, 128]", permute_1351: "f32[16, 64, 128]", permute_1352: "f32[16, 64, 128]", permute_1353: "f32[16, 128, 64]", permute_1357: "f32[1024, 1024]", permute_1362: "f32[1024, 1024]", permute_1366: "f32[1024, 1024]", div_93: "f32[1, 128, 1]", permute_1370: "f32[1024, 4096]", le_21: "b8[1, 128, 4096]", permute_1374: "f32[4096, 1024]", div_94: "f32[1, 128, 1]", permute_1378: "f32[1024, 1024]", permute_1383: "f32[16, 128, 128]", permute_1384: "f32[16, 64, 128]", permute_1385: "f32[16, 64, 128]", permute_1386: "f32[16, 128, 64]", permute_1390: "f32[1024, 1024]", permute_1395: "f32[1024, 1024]", permute_1399: "f32[1024, 1024]", div_95: "f32[1, 128, 1]", permute_1403: "f32[1024, 4096]", le_22: "b8[1, 128, 4096]", permute_1407: "f32[4096, 1024]", div_96: "f32[1, 128, 1]", permute_1411: "f32[1024, 1024]", permute_1416: "f32[16, 128, 128]", permute_1417: "f32[16, 64, 128]", permute_1418: "f32[16, 64, 128]", permute_1419: "f32[16, 128, 64]", permute_1423: "f32[1024, 1024]", permute_1428: "f32[1024, 1024]", permute_1432: "f32[1024, 1024]", div_97: "f32[1, 128, 1]", permute_1436: "f32[1024, 4096]", le_23: "b8[1, 128, 4096]", permute_1440: "f32[4096, 1024]", div_98: "f32[1, 128, 1]", permute_1444: "f32[1024, 1024]", permute_1449: "f32[16, 128, 128]", permute_1450: "f32[16, 64, 128]", permute_1451: "f32[16, 64, 128]", permute_1452: "f32[16, 128, 64]", permute_1456: "f32[1024, 1024]", permute_1461: "f32[1024, 1024]", permute_1465: "f32[1024, 1024]", div_99: "f32[1, 128, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 128112]", tangents_3: "f32[1, 16, 128, 64]", tangents_4: "f32[1, 16, 128, 64]", tangents_5: "f32[1, 16, 128, 64]", tangents_6: "f32[1, 16, 128, 64]", tangents_7: "f32[1, 16, 128, 64]", tangents_8: "f32[1, 16, 128, 64]", tangents_9: "f32[1, 16, 128, 64]", tangents_10: "f32[1, 16, 128, 64]", tangents_11: "f32[1, 16, 128, 64]", tangents_12: "f32[1, 16, 128, 64]", tangents_13: "f32[1, 16, 128, 64]", tangents_14: "f32[1, 16, 128, 64]", tangents_15: "f32[1, 16, 128, 64]", tangents_16: "f32[1, 16, 128, 64]", tangents_17: "f32[1, 16, 128, 64]", tangents_18: "f32[1, 16, 128, 64]", tangents_19: "f32[1, 16, 128, 64]", tangents_20: "f32[1, 16, 128, 64]", tangents_21: "f32[1, 16, 128, 64]", tangents_22: "f32[1, 16, 128, 64]", tangents_23: "f32[1, 16, 128, 64]", tangents_24: "f32[1, 16, 128, 64]", tangents_25: "f32[1, 16, 128, 64]", tangents_26: "f32[1, 16, 128, 64]", tangents_27: "f32[1, 16, 128, 64]", tangents_28: "f32[1, 16, 128, 64]", tangents_29: "f32[1, 16, 128, 64]", tangents_30: "f32[1, 16, 128, 64]", tangents_31: "f32[1, 16, 128, 64]", tangents_32: "f32[1, 16, 128, 64]", tangents_33: "f32[1, 16, 128, 64]", tangents_34: "f32[1, 16, 128, 64]", tangents_35: "f32[1, 16, 128, 64]", tangents_36: "f32[1, 16, 128, 64]", tangents_37: "f32[1, 16, 128, 64]", tangents_38: "f32[1, 16, 128, 64]", tangents_39: "f32[1, 16, 128, 64]", tangents_40: "f32[1, 16, 128, 64]", tangents_41: "f32[1, 16, 128, 64]", tangents_42: "f32[1, 16, 128, 64]", tangents_43: "f32[1, 16, 128, 64]", tangents_44: "f32[1, 16, 128, 64]", tangents_45: "f32[1, 16, 128, 64]", tangents_46: "f32[1, 16, 128, 64]", tangents_47: "f32[1, 16, 128, 64]", tangents_48: "f32[1, 16, 128, 64]", tangents_49: "f32[1, 16, 128, 64]", tangents_50: "f32[1, 16, 128, 64]", tangents_51: "f32[1, 128, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    sub_4: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_2, amax_1);  bmm_2 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    sub_7: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_4, amax_2);  bmm_4 = amax_2 = None
    exp_2: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    div_2: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    sub_10: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_6, amax_3);  bmm_6 = amax_3 = None
    exp_3: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    div_3: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    sub_13: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_8, amax_4);  bmm_8 = amax_4 = None
    exp_4: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    div_4: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    sub_16: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_10, amax_5);  bmm_10 = amax_5 = None
    exp_5: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    div_5: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    sub_19: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_12, amax_6);  bmm_12 = amax_6 = None
    exp_6: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    div_6: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    sub_22: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_14, amax_7);  bmm_14 = amax_7 = None
    exp_7: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    div_7: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    sub_25: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_16, amax_8);  bmm_16 = amax_8 = None
    exp_8: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    div_8: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    sub_28: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_18, amax_9);  bmm_18 = amax_9 = None
    exp_9: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    div_9: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    sub_31: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_20, amax_10);  bmm_20 = amax_10 = None
    exp_10: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    div_10: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    sub_34: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_22, amax_11);  bmm_22 = amax_11 = None
    exp_11: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    div_11: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    sub_40: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_26, amax_13);  bmm_26 = amax_13 = None
    exp_13: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    div_13: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    sub_45: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_30, amax_15);  bmm_30 = amax_15 = None
    exp_15: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    div_15: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    sub_50: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_34, amax_17);  bmm_34 = amax_17 = None
    exp_17: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_50);  sub_50 = None
    div_17: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    sub_55: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_38, amax_19);  bmm_38 = amax_19 = None
    exp_19: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    div_19: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    sub_60: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_42, amax_21);  bmm_42 = amax_21 = None
    exp_21: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    div_21: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    sub_65: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_46, amax_23);  bmm_46 = amax_23 = None
    exp_23: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
    div_23: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    sub_70: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_50, amax_25);  bmm_50 = amax_25 = None
    exp_25: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    div_25: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    sub_75: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_54, amax_27);  bmm_54 = amax_27 = None
    exp_27: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
    div_27: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    sub_80: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_58, amax_29);  bmm_58 = amax_29 = None
    exp_29: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    div_29: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
    sub_85: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_62, amax_31);  bmm_62 = amax_31 = None
    exp_31: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
    div_31: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
    sub_90: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_66, amax_33);  bmm_66 = amax_33 = None
    exp_33: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_90);  sub_90 = None
    div_33: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
    sub_95: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(bmm_70, amax_35);  bmm_70 = amax_35 = None
    exp_35: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_95);  sub_95 = None
    div_35: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1338, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_706: "i64[128]" = torch.ops.aten.reshape.default(primals_514, [-1]);  primals_514 = None
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_37: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_6);  tangents_1 = convert_element_type_6 = None
    unsqueeze_5: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_706, 1);  view_706 = None
    ne_5: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, -100)
    where_3: "i64[128, 1]" = torch.ops.aten.where.self(ne_5, unsqueeze_5, full_default_2);  unsqueeze_5 = full_default_2 = None
    full_default_5: "f32[128, 128112]" = torch.ops.aten.full.default([128, 128112], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[128, 128112]" = torch.ops.aten.scatter.value(full_default_5, 1, where_3, -1.0);  full_default_5 = where_3 = None
    where_4: "f32[128, 1]" = torch.ops.aten.where.self(ne_5, div_37, full_default_1);  ne_5 = div_37 = None
    mul_164: "f32[128, 128112]" = torch.ops.aten.mul.Tensor(scatter, where_4);  scatter = where_4 = None
    exp_37: "f32[128, 128112]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
    sum_40: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [1], True)
    mul_165: "f32[128, 128112]" = torch.ops.aten.mul.Tensor(exp_37, sum_40);  exp_37 = sum_40 = None
    sub_100: "f32[128, 128112]" = torch.ops.aten.sub.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    view_707: "f32[1, 128, 128112]" = torch.ops.aten.reshape.default(sub_100, [1, 128, 128112]);  sub_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1338, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    add_203: "f32[1, 128, 128112]" = torch.ops.aten.add.Tensor(tangents_2, view_707);  tangents_2 = view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1331, code: lm_logits = self.lm_head(outputs[0])
    view_708: "f32[128, 128112]" = torch.ops.aten.reshape.default(add_203, [128, 128112]);  add_203 = None
    permute_373: "f32[128112, 128]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_1: "f32[128112, 1024]" = torch.ops.aten.mm.default(permute_373, view_703);  permute_373 = view_703 = None
    permute_374: "f32[1024, 128112]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    mm_2: "f32[128, 1024]" = torch.ops.aten.mm.default(view_708, permute_375);  view_708 = permute_375 = None
    view_709: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 1024]);  mm_2 = None
    permute_376: "f32[128112, 1024]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1114, code: hidden_states = self.layer_norm(hidden_states)
    mul_167: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_709, primals_509);  primals_509 = None
    mul_168: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_167, 1024)
    sum_41: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_167, mul_162);  mul_167 = None
    sum_42: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_162, sum_42);  sum_42 = None
    sub_102: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_168, sum_41);  mul_168 = sum_41 = None
    sub_103: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_102, mul_170);  sub_102 = mul_170 = None
    mul_171: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_38, sub_103);  div_38 = sub_103 = None
    mul_172: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_709, mul_162);  mul_162 = None
    sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0, 1]);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_710: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_171, [128, 1024])
    mm_3: "f32[128, 4096]" = torch.ops.aten.mm.default(view_710, permute_377);  permute_377 = None
    permute_378: "f32[1024, 128]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_4: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_378, view_701);  permute_378 = view_701 = None
    permute_379: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
    sum_45: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_710, [0], True);  view_710 = None
    view_711: "f32[1024]" = torch.ops.aten.reshape.default(sum_45, [1024]);  sum_45 = None
    permute_380: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_712: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 4096]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_5: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le, full_default_1, view_712);  le = view_712 = None
    view_713: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_5, [128, 4096]);  where_5 = None
    mm_5: "f32[128, 1024]" = torch.ops.aten.mm.default(view_713, permute_381);  permute_381 = None
    permute_382: "f32[4096, 128]" = torch.ops.aten.permute.default(view_713, [1, 0])
    mm_6: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_382, view_699);  permute_382 = view_699 = None
    permute_383: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_6, [1, 0]);  mm_6 = None
    sum_46: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
    view_714: "f32[4096]" = torch.ops.aten.reshape.default(sum_46, [4096]);  sum_46 = None
    permute_384: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_715: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 1024]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_174: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_715, primals_503);  primals_503 = None
    mul_175: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_174, 1024)
    sum_47: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True)
    mul_176: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_174, mul_160);  mul_174 = None
    sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True);  mul_176 = None
    mul_177: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_160, sum_48);  sum_48 = None
    sub_105: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_175, sum_47);  mul_175 = sum_47 = None
    sub_106: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_105, mul_177);  sub_105 = mul_177 = None
    mul_178: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_39, sub_106);  div_39 = sub_106 = None
    mul_179: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_715, mul_160);  mul_160 = None
    sum_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1]);  mul_179 = None
    sum_50: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0, 1]);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_204: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_171, mul_178);  mul_171 = mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_716: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_204, [128, 1024])
    mm_7: "f32[128, 1024]" = torch.ops.aten.mm.default(view_716, permute_385);  permute_385 = None
    permute_386: "f32[1024, 128]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_8: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_386, view_697);  permute_386 = view_697 = None
    permute_387: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    sum_51: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_716, [0], True);  view_716 = None
    view_717: "f32[1024]" = torch.ops.aten.reshape.default(sum_51, [1024]);  sum_51 = None
    permute_388: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_718: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 1024]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_719: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_718, [1, 128, 16, 64]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_389: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_719, [0, 2, 1, 3]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_720: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_389, [16, 128, 64]);  permute_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_72: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_390, view_720);  permute_390 = None
    bmm_73: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_720, permute_391);  view_720 = permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_180: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_73, div_35);  bmm_73 = None
    sum_52: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [-1], True)
    mul_181: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_35, sum_52);  div_35 = sum_52 = None
    sub_107: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_74: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_392, sub_107);  permute_392 = None
    bmm_75: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_107, permute_393);  sub_107 = permute_393 = None
    permute_394: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_74, [0, 2, 1]);  bmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_721: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_72, [1, 16, 128, 64]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_205: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_50, view_721);  tangents_50 = view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_722: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_394, [1, 16, 128, 64]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_206: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_49, view_722);  tangents_49 = view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_723: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_75, [1, 16, 128, 64]);  bmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_395: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
    clone_266: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_724: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_266, [1, 128, 1024]);  clone_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_396: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_205, [0, 2, 1, 3]);  add_205 = None
    clone_267: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
    view_725: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_267, [1, 128, 1024]);  clone_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_726: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_725, [128, 1024]);  view_725 = None
    mm_9: "f32[128, 1024]" = torch.ops.aten.mm.default(view_726, permute_397);  permute_397 = None
    permute_398: "f32[1024, 128]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_10: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_398, view_267);  permute_398 = None
    permute_399: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    sum_53: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[1024]" = torch.ops.aten.reshape.default(sum_53, [1024]);  sum_53 = None
    permute_400: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    view_728: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 1024]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_207: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_51, view_728);  tangents_51 = view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_401: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_206, [0, 2, 1, 3]);  add_206 = None
    clone_268: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
    view_729: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_268, [1, 128, 1024]);  clone_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_730: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_729, [128, 1024]);  view_729 = None
    mm_11: "f32[128, 1024]" = torch.ops.aten.mm.default(view_730, permute_402);  permute_402 = None
    permute_403: "f32[1024, 128]" = torch.ops.aten.permute.default(view_730, [1, 0])
    mm_12: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_403, view_267);  permute_403 = None
    permute_404: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    sum_54: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
    view_731: "f32[1024]" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
    permute_405: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_732: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 1024]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_208: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_207, view_732);  add_207 = view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_182: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_724, 0.125);  view_724 = None
    view_733: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_182, [128, 1024]);  mul_182 = None
    mm_13: "f32[128, 1024]" = torch.ops.aten.mm.default(view_733, permute_406);  permute_406 = None
    permute_407: "f32[1024, 128]" = torch.ops.aten.permute.default(view_733, [1, 0])
    mm_14: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_407, view_683);  permute_407 = view_683 = None
    permute_408: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    sum_55: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_733, [0], True);  view_733 = None
    view_734: "f32[1024]" = torch.ops.aten.reshape.default(sum_55, [1024]);  sum_55 = None
    permute_409: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_735: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 1024]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_184: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_735, primals_493);  primals_493 = None
    mul_185: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, 1024)
    sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
    mul_186: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, mul_157);  mul_184 = None
    sum_57: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
    mul_187: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_157, sum_57);  sum_57 = None
    sub_109: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_185, sum_56);  mul_185 = sum_56 = None
    sub_110: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_109, mul_187);  sub_109 = mul_187 = None
    mul_188: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_40, sub_110);  div_40 = sub_110 = None
    mul_189: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_735, mul_157);  mul_157 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_735, [0, 1]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_209: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_204, mul_188);  add_204 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_736: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_209, [128, 1024])
    mm_15: "f32[128, 1024]" = torch.ops.aten.mm.default(view_736, permute_410);  permute_410 = None
    permute_411: "f32[1024, 128]" = torch.ops.aten.permute.default(view_736, [1, 0])
    mm_16: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_411, view_681);  permute_411 = view_681 = None
    permute_412: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_736, [0], True);  view_736 = None
    view_737: "f32[1024]" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
    permute_413: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_738: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 1024]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_739: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_738, [1, 128, 16, 64]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_414: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_739, [0, 2, 1, 3]);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_740: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_414, [16, 128, 64]);  permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_76: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_415, view_740);  permute_415 = None
    bmm_77: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_740, permute_416);  view_740 = permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_190: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_77, alias_68);  bmm_77 = None
    sum_61: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [-1], True)
    mul_191: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_68, sum_61);  alias_68 = sum_61 = None
    sub_111: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_741: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_111, [1, 16, 128, 128]);  sub_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_742: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_741, [16, 128, 128]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_78: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_417, view_742);  permute_417 = None
    bmm_79: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_742, permute_418);  view_742 = permute_418 = None
    permute_419: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_78, [0, 2, 1]);  bmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_743: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_76, [1, 16, 128, 64]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_210: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_48, view_743);  tangents_48 = view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_744: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_419, [1, 16, 128, 64]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_211: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_47, view_744);  tangents_47 = view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_745: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_79, [1, 16, 128, 64]);  bmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_420: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_745, [0, 2, 1, 3]);  view_745 = None
    clone_269: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_746: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_269, [1, 128, 1024]);  clone_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_421: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_210, [0, 2, 1, 3]);  add_210 = None
    clone_270: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_747: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_270, [1, 128, 1024]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_748: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_747, [128, 1024]);  view_747 = None
    mm_17: "f32[128, 1024]" = torch.ops.aten.mm.default(view_748, permute_422);  permute_422 = None
    permute_423: "f32[1024, 128]" = torch.ops.aten.permute.default(view_748, [1, 0])
    mm_18: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_423, view_665);  permute_423 = None
    permute_424: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_748, [0], True);  view_748 = None
    view_749: "f32[1024]" = torch.ops.aten.reshape.default(sum_62, [1024]);  sum_62 = None
    permute_425: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_750: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 1024]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_426: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_211, [0, 2, 1, 3]);  add_211 = None
    clone_271: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_751: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_271, [1, 128, 1024]);  clone_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_752: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_751, [128, 1024]);  view_751 = None
    mm_19: "f32[128, 1024]" = torch.ops.aten.mm.default(view_752, permute_427);  permute_427 = None
    permute_428: "f32[1024, 128]" = torch.ops.aten.permute.default(view_752, [1, 0])
    mm_20: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_428, view_665);  permute_428 = None
    permute_429: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_752, [0], True);  view_752 = None
    view_753: "f32[1024]" = torch.ops.aten.reshape.default(sum_63, [1024]);  sum_63 = None
    permute_430: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_754: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 1024]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_212: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_750, view_754);  view_750 = view_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_192: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_746, 0.125);  view_746 = None
    view_755: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_192, [128, 1024]);  mul_192 = None
    mm_21: "f32[128, 1024]" = torch.ops.aten.mm.default(view_755, permute_431);  permute_431 = None
    permute_432: "f32[1024, 128]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_22: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_432, view_665);  permute_432 = view_665 = None
    permute_433: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    sum_64: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[1024]" = torch.ops.aten.reshape.default(sum_64, [1024]);  sum_64 = None
    permute_434: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_757: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 1024]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_213: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_212, view_757);  add_212 = view_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_194: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_213, primals_483);  primals_483 = None
    mul_195: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_194, 1024)
    sum_65: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True)
    mul_196: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_194, mul_154);  mul_194 = None
    sum_66: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True);  mul_196 = None
    mul_197: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_154, sum_66);  sum_66 = None
    sub_113: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_195, sum_65);  mul_195 = sum_65 = None
    sub_114: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_113, mul_197);  sub_113 = mul_197 = None
    mul_198: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_41, sub_114);  div_41 = sub_114 = None
    mul_199: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_213, mul_154);  mul_154 = None
    sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
    sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 1]);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_214: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_209, mul_198);  add_209 = mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_758: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_214, [128, 1024])
    mm_23: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_435);  permute_435 = None
    permute_436: "f32[1024, 128]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_24: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_436, view_663);  permute_436 = view_663 = None
    permute_437: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_24, [1, 0]);  mm_24 = None
    sum_69: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
    view_759: "f32[1024]" = torch.ops.aten.reshape.default(sum_69, [1024]);  sum_69 = None
    permute_438: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_760: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 4096]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_6: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_1, full_default_1, view_760);  le_1 = view_760 = None
    view_761: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_6, [128, 4096]);  where_6 = None
    mm_25: "f32[128, 1024]" = torch.ops.aten.mm.default(view_761, permute_439);  permute_439 = None
    permute_440: "f32[4096, 128]" = torch.ops.aten.permute.default(view_761, [1, 0])
    mm_26: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_440, view_661);  permute_440 = view_661 = None
    permute_441: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_26, [1, 0]);  mm_26 = None
    sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_761, [0], True);  view_761 = None
    view_762: "f32[4096]" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
    permute_442: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_763: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 1024]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_201: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_763, primals_477);  primals_477 = None
    mul_202: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_201, 1024)
    sum_71: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
    mul_203: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_201, mul_152);  mul_201 = None
    sum_72: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
    mul_204: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_152, sum_72);  sum_72 = None
    sub_116: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_202, sum_71);  mul_202 = sum_71 = None
    sub_117: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_116, mul_204);  sub_116 = mul_204 = None
    mul_205: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_42, sub_117);  div_42 = sub_117 = None
    mul_206: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_763, mul_152);  mul_152 = None
    sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_763, [0, 1]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_215: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_214, mul_205);  add_214 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_764: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_215, [128, 1024])
    mm_27: "f32[128, 1024]" = torch.ops.aten.mm.default(view_764, permute_443);  permute_443 = None
    permute_444: "f32[1024, 128]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_28: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_444, view_659);  permute_444 = view_659 = None
    permute_445: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    sum_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
    view_765: "f32[1024]" = torch.ops.aten.reshape.default(sum_75, [1024]);  sum_75 = None
    permute_446: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_766: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 1024]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_767: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_766, [1, 128, 16, 64]);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_447: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_767, [0, 2, 1, 3]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_768: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_447, [16, 128, 64]);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_80: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_448, view_768);  permute_448 = None
    bmm_81: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_768, permute_449);  view_768 = permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_207: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_81, div_33);  bmm_81 = None
    sum_76: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [-1], True)
    mul_208: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_33, sum_76);  div_33 = sum_76 = None
    sub_118: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_82: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_450, sub_118);  permute_450 = None
    bmm_83: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_118, permute_451);  sub_118 = permute_451 = None
    permute_452: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_82, [0, 2, 1]);  bmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_769: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_80, [1, 16, 128, 64]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_216: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_46, view_769);  tangents_46 = view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_770: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_452, [1, 16, 128, 64]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_217: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_45, view_770);  tangents_45 = view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_771: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_83, [1, 16, 128, 64]);  bmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_453: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None
    clone_272: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_772: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_272, [1, 128, 1024]);  clone_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_454: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_216, [0, 2, 1, 3]);  add_216 = None
    clone_273: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_773: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_273, [1, 128, 1024]);  clone_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_774: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_773, [128, 1024]);  view_773 = None
    mm_29: "f32[128, 1024]" = torch.ops.aten.mm.default(view_774, permute_455);  permute_455 = None
    permute_456: "f32[1024, 128]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_30: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_456, view_267);  permute_456 = None
    permute_457: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_774, [0], True);  view_774 = None
    view_775: "f32[1024]" = torch.ops.aten.reshape.default(sum_77, [1024]);  sum_77 = None
    permute_458: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_776: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 1024]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_218: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_208, view_776);  add_208 = view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_459: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_217, [0, 2, 1, 3]);  add_217 = None
    clone_274: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_777: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_274, [1, 128, 1024]);  clone_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_778: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_777, [128, 1024]);  view_777 = None
    mm_31: "f32[128, 1024]" = torch.ops.aten.mm.default(view_778, permute_460);  permute_460 = None
    permute_461: "f32[1024, 128]" = torch.ops.aten.permute.default(view_778, [1, 0])
    mm_32: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_461, view_267);  permute_461 = None
    permute_462: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_32, [1, 0]);  mm_32 = None
    sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_778, [0], True);  view_778 = None
    view_779: "f32[1024]" = torch.ops.aten.reshape.default(sum_78, [1024]);  sum_78 = None
    permute_463: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_780: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 1024]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_219: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_218, view_780);  add_218 = view_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_209: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_772, 0.125);  view_772 = None
    view_781: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_209, [128, 1024]);  mul_209 = None
    mm_33: "f32[128, 1024]" = torch.ops.aten.mm.default(view_781, permute_464);  permute_464 = None
    permute_465: "f32[1024, 128]" = torch.ops.aten.permute.default(view_781, [1, 0])
    mm_34: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_465, view_645);  permute_465 = view_645 = None
    permute_466: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    sum_79: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_781, [0], True);  view_781 = None
    view_782: "f32[1024]" = torch.ops.aten.reshape.default(sum_79, [1024]);  sum_79 = None
    permute_467: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_783: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 1024]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_211: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_783, primals_467);  primals_467 = None
    mul_212: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_211, 1024)
    sum_80: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
    mul_213: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_211, mul_149);  mul_211 = None
    sum_81: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
    mul_214: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_149, sum_81);  sum_81 = None
    sub_120: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_212, sum_80);  mul_212 = sum_80 = None
    sub_121: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_120, mul_214);  sub_120 = mul_214 = None
    mul_215: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_43, sub_121);  div_43 = sub_121 = None
    mul_216: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_783, mul_149);  mul_149 = None
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
    sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_783, [0, 1]);  view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_220: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_215, mul_215);  add_215 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_784: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_220, [128, 1024])
    mm_35: "f32[128, 1024]" = torch.ops.aten.mm.default(view_784, permute_468);  permute_468 = None
    permute_469: "f32[1024, 128]" = torch.ops.aten.permute.default(view_784, [1, 0])
    mm_36: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_469, view_643);  permute_469 = view_643 = None
    permute_470: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    sum_84: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_784, [0], True);  view_784 = None
    view_785: "f32[1024]" = torch.ops.aten.reshape.default(sum_84, [1024]);  sum_84 = None
    permute_471: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_786: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 1024]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_787: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_786, [1, 128, 16, 64]);  view_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_472: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_788: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_472, [16, 128, 64]);  permute_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_84: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_473, view_788);  permute_473 = None
    bmm_85: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_788, permute_474);  view_788 = permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_217: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_85, alias_71);  bmm_85 = None
    sum_85: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
    mul_218: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_71, sum_85);  alias_71 = sum_85 = None
    sub_122: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_789: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_122, [1, 16, 128, 128]);  sub_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_790: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_789, [16, 128, 128]);  view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_86: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_475, view_790);  permute_475 = None
    bmm_87: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_790, permute_476);  view_790 = permute_476 = None
    permute_477: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_86, [0, 2, 1]);  bmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_791: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_84, [1, 16, 128, 64]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_221: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_44, view_791);  tangents_44 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_792: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_477, [1, 16, 128, 64]);  permute_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_222: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_43, view_792);  tangents_43 = view_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_793: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_87, [1, 16, 128, 64]);  bmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_478: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_793, [0, 2, 1, 3]);  view_793 = None
    clone_275: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
    view_794: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_275, [1, 128, 1024]);  clone_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_479: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_221, [0, 2, 1, 3]);  add_221 = None
    clone_276: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_795: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_276, [1, 128, 1024]);  clone_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_796: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_795, [128, 1024]);  view_795 = None
    mm_37: "f32[128, 1024]" = torch.ops.aten.mm.default(view_796, permute_480);  permute_480 = None
    permute_481: "f32[1024, 128]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_38: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_481, view_627);  permute_481 = None
    permute_482: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    sum_86: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[1024]" = torch.ops.aten.reshape.default(sum_86, [1024]);  sum_86 = None
    permute_483: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_798: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 1024]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_484: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_222, [0, 2, 1, 3]);  add_222 = None
    clone_277: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    view_799: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_277, [1, 128, 1024]);  clone_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_800: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_799, [128, 1024]);  view_799 = None
    mm_39: "f32[128, 1024]" = torch.ops.aten.mm.default(view_800, permute_485);  permute_485 = None
    permute_486: "f32[1024, 128]" = torch.ops.aten.permute.default(view_800, [1, 0])
    mm_40: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_486, view_627);  permute_486 = None
    permute_487: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    sum_87: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_800, [0], True);  view_800 = None
    view_801: "f32[1024]" = torch.ops.aten.reshape.default(sum_87, [1024]);  sum_87 = None
    permute_488: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_802: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 1024]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_223: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_798, view_802);  view_798 = view_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_219: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_794, 0.125);  view_794 = None
    view_803: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_219, [128, 1024]);  mul_219 = None
    mm_41: "f32[128, 1024]" = torch.ops.aten.mm.default(view_803, permute_489);  permute_489 = None
    permute_490: "f32[1024, 128]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_42: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_490, view_627);  permute_490 = view_627 = None
    permute_491: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    sum_88: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_803, [0], True);  view_803 = None
    view_804: "f32[1024]" = torch.ops.aten.reshape.default(sum_88, [1024]);  sum_88 = None
    permute_492: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
    view_805: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 1024]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_224: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_223, view_805);  add_223 = view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_221: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_224, primals_457);  primals_457 = None
    mul_222: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_221, 1024)
    sum_89: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True)
    mul_223: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_221, mul_146);  mul_221 = None
    sum_90: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
    mul_224: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_146, sum_90);  sum_90 = None
    sub_124: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_222, sum_89);  mul_222 = sum_89 = None
    sub_125: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_124, mul_224);  sub_124 = mul_224 = None
    mul_225: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_44, sub_125);  div_44 = sub_125 = None
    mul_226: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_146);  mul_146 = None
    sum_91: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_226, [0, 1]);  mul_226 = None
    sum_92: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_225: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_220, mul_225);  add_220 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_806: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_225, [128, 1024])
    mm_43: "f32[128, 4096]" = torch.ops.aten.mm.default(view_806, permute_493);  permute_493 = None
    permute_494: "f32[1024, 128]" = torch.ops.aten.permute.default(view_806, [1, 0])
    mm_44: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_494, view_625);  permute_494 = view_625 = None
    permute_495: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_806, [0], True);  view_806 = None
    view_807: "f32[1024]" = torch.ops.aten.reshape.default(sum_93, [1024]);  sum_93 = None
    permute_496: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_808: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 4096]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_7: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_2, full_default_1, view_808);  le_2 = view_808 = None
    view_809: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_7, [128, 4096]);  where_7 = None
    mm_45: "f32[128, 1024]" = torch.ops.aten.mm.default(view_809, permute_497);  permute_497 = None
    permute_498: "f32[4096, 128]" = torch.ops.aten.permute.default(view_809, [1, 0])
    mm_46: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_498, view_623);  permute_498 = view_623 = None
    permute_499: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    sum_94: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_809, [0], True);  view_809 = None
    view_810: "f32[4096]" = torch.ops.aten.reshape.default(sum_94, [4096]);  sum_94 = None
    permute_500: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_811: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 1024]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_228: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_811, primals_451);  primals_451 = None
    mul_229: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_228, 1024)
    sum_95: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True)
    mul_230: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_228, mul_144);  mul_228 = None
    sum_96: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2], True);  mul_230 = None
    mul_231: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_144, sum_96);  sum_96 = None
    sub_127: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_229, sum_95);  mul_229 = sum_95 = None
    sub_128: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_127, mul_231);  sub_127 = mul_231 = None
    mul_232: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_45, sub_128);  div_45 = sub_128 = None
    mul_233: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_811, mul_144);  mul_144 = None
    sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 1]);  mul_233 = None
    sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_811, [0, 1]);  view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_226: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_232);  add_225 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_812: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_226, [128, 1024])
    mm_47: "f32[128, 1024]" = torch.ops.aten.mm.default(view_812, permute_501);  permute_501 = None
    permute_502: "f32[1024, 128]" = torch.ops.aten.permute.default(view_812, [1, 0])
    mm_48: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_502, view_621);  permute_502 = view_621 = None
    permute_503: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_812, [0], True);  view_812 = None
    view_813: "f32[1024]" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
    permute_504: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_814: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 1024]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_815: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_814, [1, 128, 16, 64]);  view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_505: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_816: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_505, [16, 128, 64]);  permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_88: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_506, view_816);  permute_506 = None
    bmm_89: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_816, permute_507);  view_816 = permute_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_234: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_89, div_31);  bmm_89 = None
    sum_100: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [-1], True)
    mul_235: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_31, sum_100);  div_31 = sum_100 = None
    sub_129: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_90: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_508, sub_129);  permute_508 = None
    bmm_91: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_129, permute_509);  sub_129 = permute_509 = None
    permute_510: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_90, [0, 2, 1]);  bmm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_817: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_88, [1, 16, 128, 64]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_227: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_42, view_817);  tangents_42 = view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_818: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_510, [1, 16, 128, 64]);  permute_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_228: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_41, view_818);  tangents_41 = view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_819: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_91, [1, 16, 128, 64]);  bmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_511: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_819, [0, 2, 1, 3]);  view_819 = None
    clone_278: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
    view_820: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_278, [1, 128, 1024]);  clone_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_512: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_227, [0, 2, 1, 3]);  add_227 = None
    clone_279: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
    view_821: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_279, [1, 128, 1024]);  clone_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_822: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_821, [128, 1024]);  view_821 = None
    mm_49: "f32[128, 1024]" = torch.ops.aten.mm.default(view_822, permute_513);  permute_513 = None
    permute_514: "f32[1024, 128]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_50: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_514, view_267);  permute_514 = None
    permute_515: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_50, [1, 0]);  mm_50 = None
    sum_101: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[1024]" = torch.ops.aten.reshape.default(sum_101, [1024]);  sum_101 = None
    permute_516: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_824: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 1024]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_229: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_219, view_824);  add_219 = view_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_517: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_228, [0, 2, 1, 3]);  add_228 = None
    clone_280: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
    view_825: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_280, [1, 128, 1024]);  clone_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_826: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_825, [128, 1024]);  view_825 = None
    mm_51: "f32[128, 1024]" = torch.ops.aten.mm.default(view_826, permute_518);  permute_518 = None
    permute_519: "f32[1024, 128]" = torch.ops.aten.permute.default(view_826, [1, 0])
    mm_52: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_519, view_267);  permute_519 = None
    permute_520: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    sum_102: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_826, [0], True);  view_826 = None
    view_827: "f32[1024]" = torch.ops.aten.reshape.default(sum_102, [1024]);  sum_102 = None
    permute_521: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    view_828: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 1024]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_230: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_229, view_828);  add_229 = view_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_236: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_820, 0.125);  view_820 = None
    view_829: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_236, [128, 1024]);  mul_236 = None
    mm_53: "f32[128, 1024]" = torch.ops.aten.mm.default(view_829, permute_522);  permute_522 = None
    permute_523: "f32[1024, 128]" = torch.ops.aten.permute.default(view_829, [1, 0])
    mm_54: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_523, view_607);  permute_523 = view_607 = None
    permute_524: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    sum_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_829, [0], True);  view_829 = None
    view_830: "f32[1024]" = torch.ops.aten.reshape.default(sum_103, [1024]);  sum_103 = None
    permute_525: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    view_831: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 1024]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_238: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_831, primals_441);  primals_441 = None
    mul_239: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_238, 1024)
    sum_104: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_238, mul_141);  mul_238 = None
    sum_105: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_141, sum_105);  sum_105 = None
    sub_131: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_239, sum_104);  mul_239 = sum_104 = None
    sub_132: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_131, mul_241);  sub_131 = mul_241 = None
    mul_242: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_46, sub_132);  div_46 = sub_132 = None
    mul_243: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_831, mul_141);  mul_141 = None
    sum_106: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_107: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_831, [0, 1]);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_231: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_226, mul_242);  add_226 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_832: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_231, [128, 1024])
    mm_55: "f32[128, 1024]" = torch.ops.aten.mm.default(view_832, permute_526);  permute_526 = None
    permute_527: "f32[1024, 128]" = torch.ops.aten.permute.default(view_832, [1, 0])
    mm_56: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_527, view_605);  permute_527 = view_605 = None
    permute_528: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_832, [0], True);  view_832 = None
    view_833: "f32[1024]" = torch.ops.aten.reshape.default(sum_108, [1024]);  sum_108 = None
    permute_529: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_834: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 1024]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_835: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_834, [1, 128, 16, 64]);  view_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_530: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_835, [0, 2, 1, 3]);  view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_836: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_530, [16, 128, 64]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_92: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_531, view_836);  permute_531 = None
    bmm_93: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_836, permute_532);  view_836 = permute_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_244: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_93, alias_74);  bmm_93 = None
    sum_109: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [-1], True)
    mul_245: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_74, sum_109);  alias_74 = sum_109 = None
    sub_133: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_837: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_133, [1, 16, 128, 128]);  sub_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_838: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_837, [16, 128, 128]);  view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_94: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_533, view_838);  permute_533 = None
    bmm_95: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_838, permute_534);  view_838 = permute_534 = None
    permute_535: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_94, [0, 2, 1]);  bmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_839: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_92, [1, 16, 128, 64]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_232: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_40, view_839);  tangents_40 = view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_840: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_535, [1, 16, 128, 64]);  permute_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_233: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_39, view_840);  tangents_39 = view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_841: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_95, [1, 16, 128, 64]);  bmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_536: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_841, [0, 2, 1, 3]);  view_841 = None
    clone_281: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_536, memory_format = torch.contiguous_format);  permute_536 = None
    view_842: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_281, [1, 128, 1024]);  clone_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_537: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_232, [0, 2, 1, 3]);  add_232 = None
    clone_282: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
    view_843: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_282, [1, 128, 1024]);  clone_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_844: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_843, [128, 1024]);  view_843 = None
    mm_57: "f32[128, 1024]" = torch.ops.aten.mm.default(view_844, permute_538);  permute_538 = None
    permute_539: "f32[1024, 128]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_58: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_539, view_589);  permute_539 = None
    permute_540: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    sum_110: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
    view_845: "f32[1024]" = torch.ops.aten.reshape.default(sum_110, [1024]);  sum_110 = None
    permute_541: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    view_846: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 1024]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_542: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_233, [0, 2, 1, 3]);  add_233 = None
    clone_283: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_542, memory_format = torch.contiguous_format);  permute_542 = None
    view_847: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_283, [1, 128, 1024]);  clone_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_848: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_847, [128, 1024]);  view_847 = None
    mm_59: "f32[128, 1024]" = torch.ops.aten.mm.default(view_848, permute_543);  permute_543 = None
    permute_544: "f32[1024, 128]" = torch.ops.aten.permute.default(view_848, [1, 0])
    mm_60: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_544, view_589);  permute_544 = None
    permute_545: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    sum_111: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True);  view_848 = None
    view_849: "f32[1024]" = torch.ops.aten.reshape.default(sum_111, [1024]);  sum_111 = None
    permute_546: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_545, [1, 0]);  permute_545 = None
    view_850: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 1024]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_234: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_846, view_850);  view_846 = view_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_246: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_842, 0.125);  view_842 = None
    view_851: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_246, [128, 1024]);  mul_246 = None
    mm_61: "f32[128, 1024]" = torch.ops.aten.mm.default(view_851, permute_547);  permute_547 = None
    permute_548: "f32[1024, 128]" = torch.ops.aten.permute.default(view_851, [1, 0])
    mm_62: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_548, view_589);  permute_548 = view_589 = None
    permute_549: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_112: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
    view_852: "f32[1024]" = torch.ops.aten.reshape.default(sum_112, [1024]);  sum_112 = None
    permute_550: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_549, [1, 0]);  permute_549 = None
    view_853: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 1024]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_235: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_234, view_853);  add_234 = view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_248: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_235, primals_431);  primals_431 = None
    mul_249: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_248, 1024)
    sum_113: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_248, mul_138);  mul_248 = None
    sum_114: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, sum_114);  sum_114 = None
    sub_135: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_249, sum_113);  mul_249 = sum_113 = None
    sub_136: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_135, mul_251);  sub_135 = mul_251 = None
    mul_252: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_47, sub_136);  div_47 = sub_136 = None
    mul_253: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_235, mul_138);  mul_138 = None
    sum_115: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_116: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_235, [0, 1]);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_236: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_231, mul_252);  add_231 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_854: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_236, [128, 1024])
    mm_63: "f32[128, 4096]" = torch.ops.aten.mm.default(view_854, permute_551);  permute_551 = None
    permute_552: "f32[1024, 128]" = torch.ops.aten.permute.default(view_854, [1, 0])
    mm_64: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_552, view_587);  permute_552 = view_587 = None
    permute_553: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_117: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_854, [0], True);  view_854 = None
    view_855: "f32[1024]" = torch.ops.aten.reshape.default(sum_117, [1024]);  sum_117 = None
    permute_554: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
    view_856: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 4096]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_8: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_3, full_default_1, view_856);  le_3 = view_856 = None
    view_857: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_8, [128, 4096]);  where_8 = None
    mm_65: "f32[128, 1024]" = torch.ops.aten.mm.default(view_857, permute_555);  permute_555 = None
    permute_556: "f32[4096, 128]" = torch.ops.aten.permute.default(view_857, [1, 0])
    mm_66: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_556, view_585);  permute_556 = view_585 = None
    permute_557: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_118: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_857, [0], True);  view_857 = None
    view_858: "f32[4096]" = torch.ops.aten.reshape.default(sum_118, [4096]);  sum_118 = None
    permute_558: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    view_859: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 1024]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_255: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_859, primals_425);  primals_425 = None
    mul_256: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_255, 1024)
    sum_119: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_255, mul_136);  mul_255 = None
    sum_120: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_136, sum_120);  sum_120 = None
    sub_138: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_256, sum_119);  mul_256 = sum_119 = None
    sub_139: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_138, mul_258);  sub_138 = mul_258 = None
    mul_259: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_48, sub_139);  div_48 = sub_139 = None
    mul_260: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_859, mul_136);  mul_136 = None
    sum_121: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_122: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_859, [0, 1]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_237: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_236, mul_259);  add_236 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_860: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_237, [128, 1024])
    mm_67: "f32[128, 1024]" = torch.ops.aten.mm.default(view_860, permute_559);  permute_559 = None
    permute_560: "f32[1024, 128]" = torch.ops.aten.permute.default(view_860, [1, 0])
    mm_68: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_560, view_583);  permute_560 = view_583 = None
    permute_561: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_860, [0], True);  view_860 = None
    view_861: "f32[1024]" = torch.ops.aten.reshape.default(sum_123, [1024]);  sum_123 = None
    permute_562: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    view_862: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 1024]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_863: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_862, [1, 128, 16, 64]);  view_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_563: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_863, [0, 2, 1, 3]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_864: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_563, [16, 128, 64]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_96: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_564, view_864);  permute_564 = None
    bmm_97: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_864, permute_565);  view_864 = permute_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_261: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_97, div_29);  bmm_97 = None
    sum_124: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [-1], True)
    mul_262: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_29, sum_124);  div_29 = sum_124 = None
    sub_140: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_98: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_566, sub_140);  permute_566 = None
    bmm_99: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_140, permute_567);  sub_140 = permute_567 = None
    permute_568: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_98, [0, 2, 1]);  bmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_865: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_96, [1, 16, 128, 64]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_238: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_38, view_865);  tangents_38 = view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_866: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_568, [1, 16, 128, 64]);  permute_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_239: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_37, view_866);  tangents_37 = view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_867: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_99, [1, 16, 128, 64]);  bmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_569: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_867, [0, 2, 1, 3]);  view_867 = None
    clone_284: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_569, memory_format = torch.contiguous_format);  permute_569 = None
    view_868: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_284, [1, 128, 1024]);  clone_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_570: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_238, [0, 2, 1, 3]);  add_238 = None
    clone_285: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
    view_869: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_285, [1, 128, 1024]);  clone_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_870: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_869, [128, 1024]);  view_869 = None
    mm_69: "f32[128, 1024]" = torch.ops.aten.mm.default(view_870, permute_571);  permute_571 = None
    permute_572: "f32[1024, 128]" = torch.ops.aten.permute.default(view_870, [1, 0])
    mm_70: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_572, view_267);  permute_572 = None
    permute_573: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    sum_125: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
    view_871: "f32[1024]" = torch.ops.aten.reshape.default(sum_125, [1024]);  sum_125 = None
    permute_574: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_872: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 1024]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_240: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_230, view_872);  add_230 = view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_575: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_239, [0, 2, 1, 3]);  add_239 = None
    clone_286: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
    view_873: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_286, [1, 128, 1024]);  clone_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_874: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_873, [128, 1024]);  view_873 = None
    mm_71: "f32[128, 1024]" = torch.ops.aten.mm.default(view_874, permute_576);  permute_576 = None
    permute_577: "f32[1024, 128]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_72: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_577, view_267);  permute_577 = None
    permute_578: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    sum_126: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[1024]" = torch.ops.aten.reshape.default(sum_126, [1024]);  sum_126 = None
    permute_579: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
    view_876: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 1024]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_241: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_240, view_876);  add_240 = view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_263: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_868, 0.125);  view_868 = None
    view_877: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_263, [128, 1024]);  mul_263 = None
    mm_73: "f32[128, 1024]" = torch.ops.aten.mm.default(view_877, permute_580);  permute_580 = None
    permute_581: "f32[1024, 128]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_74: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_581, view_569);  permute_581 = view_569 = None
    permute_582: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    sum_127: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[1024]" = torch.ops.aten.reshape.default(sum_127, [1024]);  sum_127 = None
    permute_583: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
    view_879: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 1024]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_265: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_879, primals_415);  primals_415 = None
    mul_266: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_265, 1024)
    sum_128: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True)
    mul_267: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_265, mul_133);  mul_265 = None
    sum_129: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True);  mul_267 = None
    mul_268: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_133, sum_129);  sum_129 = None
    sub_142: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_266, sum_128);  mul_266 = sum_128 = None
    sub_143: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_142, mul_268);  sub_142 = mul_268 = None
    mul_269: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_49, sub_143);  div_49 = sub_143 = None
    mul_270: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_879, mul_133);  mul_133 = None
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1]);  mul_270 = None
    sum_131: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_879, [0, 1]);  view_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_242: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_269);  add_237 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_880: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_242, [128, 1024])
    mm_75: "f32[128, 1024]" = torch.ops.aten.mm.default(view_880, permute_584);  permute_584 = None
    permute_585: "f32[1024, 128]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_76: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_585, view_567);  permute_585 = view_567 = None
    permute_586: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_132: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[1024]" = torch.ops.aten.reshape.default(sum_132, [1024]);  sum_132 = None
    permute_587: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    view_882: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 1024]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_883: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_882, [1, 128, 16, 64]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_588: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_883, [0, 2, 1, 3]);  view_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_884: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_588, [16, 128, 64]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_100: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_589, view_884);  permute_589 = None
    bmm_101: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_884, permute_590);  view_884 = permute_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_271: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_101, alias_77);  bmm_101 = None
    sum_133: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [-1], True)
    mul_272: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_77, sum_133);  alias_77 = sum_133 = None
    sub_144: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_885: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_144, [1, 16, 128, 128]);  sub_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_886: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_885, [16, 128, 128]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_102: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_591, view_886);  permute_591 = None
    bmm_103: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_886, permute_592);  view_886 = permute_592 = None
    permute_593: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_102, [0, 2, 1]);  bmm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_887: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_100, [1, 16, 128, 64]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_243: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_36, view_887);  tangents_36 = view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_888: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_593, [1, 16, 128, 64]);  permute_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_244: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_35, view_888);  tangents_35 = view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_889: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_103, [1, 16, 128, 64]);  bmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_594: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_889, [0, 2, 1, 3]);  view_889 = None
    clone_287: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_594, memory_format = torch.contiguous_format);  permute_594 = None
    view_890: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_287, [1, 128, 1024]);  clone_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_595: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_243, [0, 2, 1, 3]);  add_243 = None
    clone_288: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_595, memory_format = torch.contiguous_format);  permute_595 = None
    view_891: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_288, [1, 128, 1024]);  clone_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_892: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_891, [128, 1024]);  view_891 = None
    mm_77: "f32[128, 1024]" = torch.ops.aten.mm.default(view_892, permute_596);  permute_596 = None
    permute_597: "f32[1024, 128]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_78: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_597, view_551);  permute_597 = None
    permute_598: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_134: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[1024]" = torch.ops.aten.reshape.default(sum_134, [1024]);  sum_134 = None
    permute_599: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_894: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 1024]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_600: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_244, [0, 2, 1, 3]);  add_244 = None
    clone_289: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_600, memory_format = torch.contiguous_format);  permute_600 = None
    view_895: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_289, [1, 128, 1024]);  clone_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_896: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_895, [128, 1024]);  view_895 = None
    mm_79: "f32[128, 1024]" = torch.ops.aten.mm.default(view_896, permute_601);  permute_601 = None
    permute_602: "f32[1024, 128]" = torch.ops.aten.permute.default(view_896, [1, 0])
    mm_80: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_602, view_551);  permute_602 = None
    permute_603: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_135: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_896, [0], True);  view_896 = None
    view_897: "f32[1024]" = torch.ops.aten.reshape.default(sum_135, [1024]);  sum_135 = None
    permute_604: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    view_898: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 1024]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_245: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_894, view_898);  view_894 = view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_273: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_890, 0.125);  view_890 = None
    view_899: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_273, [128, 1024]);  mul_273 = None
    mm_81: "f32[128, 1024]" = torch.ops.aten.mm.default(view_899, permute_605);  permute_605 = None
    permute_606: "f32[1024, 128]" = torch.ops.aten.permute.default(view_899, [1, 0])
    mm_82: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_606, view_551);  permute_606 = view_551 = None
    permute_607: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    sum_136: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_899, [0], True);  view_899 = None
    view_900: "f32[1024]" = torch.ops.aten.reshape.default(sum_136, [1024]);  sum_136 = None
    permute_608: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_901: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 1024]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_246: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_245, view_901);  add_245 = view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_275: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_246, primals_405);  primals_405 = None
    mul_276: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_275, 1024)
    sum_137: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True)
    mul_277: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_275, mul_130);  mul_275 = None
    sum_138: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
    mul_278: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_130, sum_138);  sum_138 = None
    sub_146: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_276, sum_137);  mul_276 = sum_137 = None
    sub_147: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_146, mul_278);  sub_146 = mul_278 = None
    mul_279: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_147);  div_50 = sub_147 = None
    mul_280: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_246, mul_130);  mul_130 = None
    sum_139: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1]);  mul_280 = None
    sum_140: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_246, [0, 1]);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_247: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_242, mul_279);  add_242 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_902: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_247, [128, 1024])
    mm_83: "f32[128, 4096]" = torch.ops.aten.mm.default(view_902, permute_609);  permute_609 = None
    permute_610: "f32[1024, 128]" = torch.ops.aten.permute.default(view_902, [1, 0])
    mm_84: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_610, view_549);  permute_610 = view_549 = None
    permute_611: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_141: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
    view_903: "f32[1024]" = torch.ops.aten.reshape.default(sum_141, [1024]);  sum_141 = None
    permute_612: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    view_904: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 4096]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_9: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_4, full_default_1, view_904);  le_4 = view_904 = None
    view_905: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_9, [128, 4096]);  where_9 = None
    mm_85: "f32[128, 1024]" = torch.ops.aten.mm.default(view_905, permute_613);  permute_613 = None
    permute_614: "f32[4096, 128]" = torch.ops.aten.permute.default(view_905, [1, 0])
    mm_86: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_614, view_547);  permute_614 = view_547 = None
    permute_615: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_142: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
    view_906: "f32[4096]" = torch.ops.aten.reshape.default(sum_142, [4096]);  sum_142 = None
    permute_616: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
    view_907: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 1024]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_282: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_907, primals_399);  primals_399 = None
    mul_283: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_282, 1024)
    sum_143: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_282, mul_128);  mul_282 = None
    sum_144: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, sum_144);  sum_144 = None
    sub_149: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_283, sum_143);  mul_283 = sum_143 = None
    sub_150: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_149, mul_285);  sub_149 = mul_285 = None
    mul_286: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_150);  div_51 = sub_150 = None
    mul_287: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_907, mul_128);  mul_128 = None
    sum_145: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_146: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_907, [0, 1]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_248: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_247, mul_286);  add_247 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_908: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_248, [128, 1024])
    mm_87: "f32[128, 1024]" = torch.ops.aten.mm.default(view_908, permute_617);  permute_617 = None
    permute_618: "f32[1024, 128]" = torch.ops.aten.permute.default(view_908, [1, 0])
    mm_88: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_618, view_545);  permute_618 = view_545 = None
    permute_619: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_147: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
    view_909: "f32[1024]" = torch.ops.aten.reshape.default(sum_147, [1024]);  sum_147 = None
    permute_620: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    view_910: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 1024]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_911: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_910, [1, 128, 16, 64]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_621: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_911, [0, 2, 1, 3]);  view_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_912: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_621, [16, 128, 64]);  permute_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_104: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_622, view_912);  permute_622 = None
    bmm_105: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_912, permute_623);  view_912 = permute_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_288: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_105, div_27);  bmm_105 = None
    sum_148: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [-1], True)
    mul_289: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_27, sum_148);  div_27 = sum_148 = None
    sub_151: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_106: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_624, sub_151);  permute_624 = None
    bmm_107: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_151, permute_625);  sub_151 = permute_625 = None
    permute_626: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_106, [0, 2, 1]);  bmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_913: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_104, [1, 16, 128, 64]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_249: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_34, view_913);  tangents_34 = view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_914: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_626, [1, 16, 128, 64]);  permute_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_250: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_33, view_914);  tangents_33 = view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_915: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_107, [1, 16, 128, 64]);  bmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_627: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_915, [0, 2, 1, 3]);  view_915 = None
    clone_290: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
    view_916: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_290, [1, 128, 1024]);  clone_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_628: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_249, [0, 2, 1, 3]);  add_249 = None
    clone_291: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
    view_917: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_291, [1, 128, 1024]);  clone_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_918: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_917, [128, 1024]);  view_917 = None
    mm_89: "f32[128, 1024]" = torch.ops.aten.mm.default(view_918, permute_629);  permute_629 = None
    permute_630: "f32[1024, 128]" = torch.ops.aten.permute.default(view_918, [1, 0])
    mm_90: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_630, view_267);  permute_630 = None
    permute_631: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_149: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_918, [0], True);  view_918 = None
    view_919: "f32[1024]" = torch.ops.aten.reshape.default(sum_149, [1024]);  sum_149 = None
    permute_632: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_920: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 1024]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_251: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_241, view_920);  add_241 = view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_633: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_250, [0, 2, 1, 3]);  add_250 = None
    clone_292: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_633, memory_format = torch.contiguous_format);  permute_633 = None
    view_921: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_292, [1, 128, 1024]);  clone_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_922: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_921, [128, 1024]);  view_921 = None
    mm_91: "f32[128, 1024]" = torch.ops.aten.mm.default(view_922, permute_634);  permute_634 = None
    permute_635: "f32[1024, 128]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_92: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_635, view_267);  permute_635 = None
    permute_636: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[1024]" = torch.ops.aten.reshape.default(sum_150, [1024]);  sum_150 = None
    permute_637: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    view_924: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 1024]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_252: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_251, view_924);  add_251 = view_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_290: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_916, 0.125);  view_916 = None
    view_925: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_290, [128, 1024]);  mul_290 = None
    mm_93: "f32[128, 1024]" = torch.ops.aten.mm.default(view_925, permute_638);  permute_638 = None
    permute_639: "f32[1024, 128]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_94: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_639, view_531);  permute_639 = view_531 = None
    permute_640: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    sum_151: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
    view_926: "f32[1024]" = torch.ops.aten.reshape.default(sum_151, [1024]);  sum_151 = None
    permute_641: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    view_927: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 1024]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_292: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_927, primals_389);  primals_389 = None
    mul_293: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_292, 1024)
    sum_152: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
    mul_294: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_292, mul_125);  mul_292 = None
    sum_153: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
    mul_295: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_125, sum_153);  sum_153 = None
    sub_153: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_293, sum_152);  mul_293 = sum_152 = None
    sub_154: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_153, mul_295);  sub_153 = mul_295 = None
    mul_296: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_154);  div_52 = sub_154 = None
    mul_297: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_927, mul_125);  mul_125 = None
    sum_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
    sum_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_927, [0, 1]);  view_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_253: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_248, mul_296);  add_248 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_928: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_253, [128, 1024])
    mm_95: "f32[128, 1024]" = torch.ops.aten.mm.default(view_928, permute_642);  permute_642 = None
    permute_643: "f32[1024, 128]" = torch.ops.aten.permute.default(view_928, [1, 0])
    mm_96: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_643, view_529);  permute_643 = view_529 = None
    permute_644: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    sum_156: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_928, [0], True);  view_928 = None
    view_929: "f32[1024]" = torch.ops.aten.reshape.default(sum_156, [1024]);  sum_156 = None
    permute_645: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    view_930: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 1024]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_931: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_930, [1, 128, 16, 64]);  view_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_646: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_932: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_646, [16, 128, 64]);  permute_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_108: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_647, view_932);  permute_647 = None
    bmm_109: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_932, permute_648);  view_932 = permute_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_298: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_109, alias_80);  bmm_109 = None
    sum_157: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [-1], True)
    mul_299: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_80, sum_157);  alias_80 = sum_157 = None
    sub_155: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_933: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_155, [1, 16, 128, 128]);  sub_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_934: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_933, [16, 128, 128]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_110: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_649, view_934);  permute_649 = None
    bmm_111: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_934, permute_650);  view_934 = permute_650 = None
    permute_651: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_110, [0, 2, 1]);  bmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_935: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_108, [1, 16, 128, 64]);  bmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_254: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_32, view_935);  tangents_32 = view_935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_936: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_651, [1, 16, 128, 64]);  permute_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_255: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_31, view_936);  tangents_31 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_937: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_111, [1, 16, 128, 64]);  bmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_652: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_937, [0, 2, 1, 3]);  view_937 = None
    clone_293: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
    view_938: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_293, [1, 128, 1024]);  clone_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_653: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_254, [0, 2, 1, 3]);  add_254 = None
    clone_294: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
    view_939: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_294, [1, 128, 1024]);  clone_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_940: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_939, [128, 1024]);  view_939 = None
    mm_97: "f32[128, 1024]" = torch.ops.aten.mm.default(view_940, permute_654);  permute_654 = None
    permute_655: "f32[1024, 128]" = torch.ops.aten.permute.default(view_940, [1, 0])
    mm_98: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_655, view_513);  permute_655 = None
    permute_656: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_98, [1, 0]);  mm_98 = None
    sum_158: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
    view_941: "f32[1024]" = torch.ops.aten.reshape.default(sum_158, [1024]);  sum_158 = None
    permute_657: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    view_942: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 1024]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_658: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_255, [0, 2, 1, 3]);  add_255 = None
    clone_295: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_658, memory_format = torch.contiguous_format);  permute_658 = None
    view_943: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_295, [1, 128, 1024]);  clone_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_944: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_943, [128, 1024]);  view_943 = None
    mm_99: "f32[128, 1024]" = torch.ops.aten.mm.default(view_944, permute_659);  permute_659 = None
    permute_660: "f32[1024, 128]" = torch.ops.aten.permute.default(view_944, [1, 0])
    mm_100: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_660, view_513);  permute_660 = None
    permute_661: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_159: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_944, [0], True);  view_944 = None
    view_945: "f32[1024]" = torch.ops.aten.reshape.default(sum_159, [1024]);  sum_159 = None
    permute_662: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    view_946: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 1024]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_256: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_942, view_946);  view_942 = view_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_300: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_938, 0.125);  view_938 = None
    view_947: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_300, [128, 1024]);  mul_300 = None
    mm_101: "f32[128, 1024]" = torch.ops.aten.mm.default(view_947, permute_663);  permute_663 = None
    permute_664: "f32[1024, 128]" = torch.ops.aten.permute.default(view_947, [1, 0])
    mm_102: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_664, view_513);  permute_664 = view_513 = None
    permute_665: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_160: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_947, [0], True);  view_947 = None
    view_948: "f32[1024]" = torch.ops.aten.reshape.default(sum_160, [1024]);  sum_160 = None
    permute_666: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    view_949: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 1024]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_257: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_256, view_949);  add_256 = view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_302: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_257, primals_379);  primals_379 = None
    mul_303: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_302, 1024)
    sum_161: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
    mul_304: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_302, mul_122);  mul_302 = None
    sum_162: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
    mul_305: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_122, sum_162);  sum_162 = None
    sub_157: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_303, sum_161);  mul_303 = sum_161 = None
    sub_158: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_157, mul_305);  sub_157 = mul_305 = None
    mul_306: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_53, sub_158);  div_53 = sub_158 = None
    mul_307: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_257, mul_122);  mul_122 = None
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_258: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_253, mul_306);  add_253 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_950: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_258, [128, 1024])
    mm_103: "f32[128, 4096]" = torch.ops.aten.mm.default(view_950, permute_667);  permute_667 = None
    permute_668: "f32[1024, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_104: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_668, view_511);  permute_668 = view_511 = None
    permute_669: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_165: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
    view_951: "f32[1024]" = torch.ops.aten.reshape.default(sum_165, [1024]);  sum_165 = None
    permute_670: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_952: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 4096]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_10: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_5, full_default_1, view_952);  le_5 = view_952 = None
    view_953: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_10, [128, 4096]);  where_10 = None
    mm_105: "f32[128, 1024]" = torch.ops.aten.mm.default(view_953, permute_671);  permute_671 = None
    permute_672: "f32[4096, 128]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_106: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_672, view_509);  permute_672 = view_509 = None
    permute_673: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_166: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
    view_954: "f32[4096]" = torch.ops.aten.reshape.default(sum_166, [4096]);  sum_166 = None
    permute_674: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_955: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 1024]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_309: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_955, primals_373);  primals_373 = None
    mul_310: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_309, 1024)
    sum_167: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_309, mul_120);  mul_309 = None
    sum_168: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, sum_168);  sum_168 = None
    sub_160: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_310, sum_167);  mul_310 = sum_167 = None
    sub_161: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_160, mul_312);  sub_160 = mul_312 = None
    mul_313: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_161);  div_54 = sub_161 = None
    mul_314: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_955, mul_120);  mul_120 = None
    sum_169: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_170: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_955, [0, 1]);  view_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_259: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_258, mul_313);  add_258 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_956: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_259, [128, 1024])
    mm_107: "f32[128, 1024]" = torch.ops.aten.mm.default(view_956, permute_675);  permute_675 = None
    permute_676: "f32[1024, 128]" = torch.ops.aten.permute.default(view_956, [1, 0])
    mm_108: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_676, view_507);  permute_676 = view_507 = None
    permute_677: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_171: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_956, [0], True);  view_956 = None
    view_957: "f32[1024]" = torch.ops.aten.reshape.default(sum_171, [1024]);  sum_171 = None
    permute_678: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_958: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 1024]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_959: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_958, [1, 128, 16, 64]);  view_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_679: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_959, [0, 2, 1, 3]);  view_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_960: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_679, [16, 128, 64]);  permute_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_112: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_680, view_960);  permute_680 = None
    bmm_113: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_960, permute_681);  view_960 = permute_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_315: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_113, div_25);  bmm_113 = None
    sum_172: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [-1], True)
    mul_316: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_25, sum_172);  div_25 = sum_172 = None
    sub_162: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_114: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_682, sub_162);  permute_682 = None
    bmm_115: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_162, permute_683);  sub_162 = permute_683 = None
    permute_684: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_114, [0, 2, 1]);  bmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_961: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_112, [1, 16, 128, 64]);  bmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_260: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_30, view_961);  tangents_30 = view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_962: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_684, [1, 16, 128, 64]);  permute_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_261: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_29, view_962);  tangents_29 = view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_963: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_115, [1, 16, 128, 64]);  bmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_685: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_963, [0, 2, 1, 3]);  view_963 = None
    clone_296: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_964: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_296, [1, 128, 1024]);  clone_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_686: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_260, [0, 2, 1, 3]);  add_260 = None
    clone_297: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
    view_965: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_297, [1, 128, 1024]);  clone_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_966: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_965, [128, 1024]);  view_965 = None
    mm_109: "f32[128, 1024]" = torch.ops.aten.mm.default(view_966, permute_687);  permute_687 = None
    permute_688: "f32[1024, 128]" = torch.ops.aten.permute.default(view_966, [1, 0])
    mm_110: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_688, view_267);  permute_688 = None
    permute_689: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_173: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_966, [0], True);  view_966 = None
    view_967: "f32[1024]" = torch.ops.aten.reshape.default(sum_173, [1024]);  sum_173 = None
    permute_690: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_689, [1, 0]);  permute_689 = None
    view_968: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 1024]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_262: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_252, view_968);  add_252 = view_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_691: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_261, [0, 2, 1, 3]);  add_261 = None
    clone_298: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_691, memory_format = torch.contiguous_format);  permute_691 = None
    view_969: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_298, [1, 128, 1024]);  clone_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_970: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_969, [128, 1024]);  view_969 = None
    mm_111: "f32[128, 1024]" = torch.ops.aten.mm.default(view_970, permute_692);  permute_692 = None
    permute_693: "f32[1024, 128]" = torch.ops.aten.permute.default(view_970, [1, 0])
    mm_112: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_693, view_267);  permute_693 = None
    permute_694: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_174: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_970, [0], True);  view_970 = None
    view_971: "f32[1024]" = torch.ops.aten.reshape.default(sum_174, [1024]);  sum_174 = None
    permute_695: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    view_972: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 1024]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_263: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_262, view_972);  add_262 = view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_317: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_964, 0.125);  view_964 = None
    view_973: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_317, [128, 1024]);  mul_317 = None
    mm_113: "f32[128, 1024]" = torch.ops.aten.mm.default(view_973, permute_696);  permute_696 = None
    permute_697: "f32[1024, 128]" = torch.ops.aten.permute.default(view_973, [1, 0])
    mm_114: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_697, view_493);  permute_697 = view_493 = None
    permute_698: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_175: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_973, [0], True);  view_973 = None
    view_974: "f32[1024]" = torch.ops.aten.reshape.default(sum_175, [1024]);  sum_175 = None
    permute_699: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_975: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_113, [1, 128, 1024]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_319: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_975, primals_363);  primals_363 = None
    mul_320: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_319, 1024)
    sum_176: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_319, mul_117);  mul_319 = None
    sum_177: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_117, sum_177);  sum_177 = None
    sub_164: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_320, sum_176);  mul_320 = sum_176 = None
    sub_165: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_164, mul_322);  sub_164 = mul_322 = None
    mul_323: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_165);  div_55 = sub_165 = None
    mul_324: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_975, mul_117);  mul_117 = None
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_975, [0, 1]);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_264: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_259, mul_323);  add_259 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_976: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_264, [128, 1024])
    mm_115: "f32[128, 1024]" = torch.ops.aten.mm.default(view_976, permute_700);  permute_700 = None
    permute_701: "f32[1024, 128]" = torch.ops.aten.permute.default(view_976, [1, 0])
    mm_116: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_701, view_491);  permute_701 = view_491 = None
    permute_702: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
    view_977: "f32[1024]" = torch.ops.aten.reshape.default(sum_180, [1024]);  sum_180 = None
    permute_703: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_978: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_115, [1, 128, 1024]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_979: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_978, [1, 128, 16, 64]);  view_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_704: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_979, [0, 2, 1, 3]);  view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_980: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_704, [16, 128, 64]);  permute_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_116: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_705, view_980);  permute_705 = None
    bmm_117: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_980, permute_706);  view_980 = permute_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_325: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_117, alias_83);  bmm_117 = None
    sum_181: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [-1], True)
    mul_326: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_83, sum_181);  alias_83 = sum_181 = None
    sub_166: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_981: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_166, [1, 16, 128, 128]);  sub_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_982: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_981, [16, 128, 128]);  view_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_118: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_707, view_982);  permute_707 = None
    bmm_119: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_982, permute_708);  view_982 = permute_708 = None
    permute_709: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_118, [0, 2, 1]);  bmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_983: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_116, [1, 16, 128, 64]);  bmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_265: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_28, view_983);  tangents_28 = view_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_984: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_709, [1, 16, 128, 64]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_266: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_27, view_984);  tangents_27 = view_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_985: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_119, [1, 16, 128, 64]);  bmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_710: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None
    clone_299: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_710, memory_format = torch.contiguous_format);  permute_710 = None
    view_986: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_299, [1, 128, 1024]);  clone_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_711: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_265, [0, 2, 1, 3]);  add_265 = None
    clone_300: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_711, memory_format = torch.contiguous_format);  permute_711 = None
    view_987: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_300, [1, 128, 1024]);  clone_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_988: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_987, [128, 1024]);  view_987 = None
    mm_117: "f32[128, 1024]" = torch.ops.aten.mm.default(view_988, permute_712);  permute_712 = None
    permute_713: "f32[1024, 128]" = torch.ops.aten.permute.default(view_988, [1, 0])
    mm_118: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_713, view_475);  permute_713 = None
    permute_714: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_988, [0], True);  view_988 = None
    view_989: "f32[1024]" = torch.ops.aten.reshape.default(sum_182, [1024]);  sum_182 = None
    permute_715: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
    view_990: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_117, [1, 128, 1024]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_716: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_266, [0, 2, 1, 3]);  add_266 = None
    clone_301: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_991: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_301, [1, 128, 1024]);  clone_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_992: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_991, [128, 1024]);  view_991 = None
    mm_119: "f32[128, 1024]" = torch.ops.aten.mm.default(view_992, permute_717);  permute_717 = None
    permute_718: "f32[1024, 128]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_120: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_718, view_475);  permute_718 = None
    permute_719: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[1024]" = torch.ops.aten.reshape.default(sum_183, [1024]);  sum_183 = None
    permute_720: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
    view_994: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 1024]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_267: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_990, view_994);  view_990 = view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_327: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_986, 0.125);  view_986 = None
    view_995: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_327, [128, 1024]);  mul_327 = None
    mm_121: "f32[128, 1024]" = torch.ops.aten.mm.default(view_995, permute_721);  permute_721 = None
    permute_722: "f32[1024, 128]" = torch.ops.aten.permute.default(view_995, [1, 0])
    mm_122: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_722, view_475);  permute_722 = view_475 = None
    permute_723: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    sum_184: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
    view_996: "f32[1024]" = torch.ops.aten.reshape.default(sum_184, [1024]);  sum_184 = None
    permute_724: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_997: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 1024]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_268: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_267, view_997);  add_267 = view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_329: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_268, primals_353);  primals_353 = None
    mul_330: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_329, 1024)
    sum_185: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_329, mul_114);  mul_329 = None
    sum_186: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, sum_186);  sum_186 = None
    sub_168: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_330, sum_185);  mul_330 = sum_185 = None
    sub_169: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_168, mul_332);  sub_168 = mul_332 = None
    mul_333: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_56, sub_169);  div_56 = sub_169 = None
    mul_334: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_268, mul_114);  mul_114 = None
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_268, [0, 1]);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_269: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_264, mul_333);  add_264 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_998: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_269, [128, 1024])
    mm_123: "f32[128, 4096]" = torch.ops.aten.mm.default(view_998, permute_725);  permute_725 = None
    permute_726: "f32[1024, 128]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_124: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_726, view_473);  permute_726 = view_473 = None
    permute_727: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_189: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True);  view_998 = None
    view_999: "f32[1024]" = torch.ops.aten.reshape.default(sum_189, [1024]);  sum_189 = None
    permute_728: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_1000: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_11: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_6, full_default_1, view_1000);  le_6 = view_1000 = None
    view_1001: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_11, [128, 4096]);  where_11 = None
    mm_125: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1001, permute_729);  permute_729 = None
    permute_730: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1001, [1, 0])
    mm_126: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_730, view_471);  permute_730 = view_471 = None
    permute_731: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_190: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1001, [0], True);  view_1001 = None
    view_1002: "f32[4096]" = torch.ops.aten.reshape.default(sum_190, [4096]);  sum_190 = None
    permute_732: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_1003: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 1024]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_336: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1003, primals_347);  primals_347 = None
    mul_337: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_336, 1024)
    sum_191: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True)
    mul_338: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_336, mul_112);  mul_336 = None
    sum_192: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True);  mul_338 = None
    mul_339: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_112, sum_192);  sum_192 = None
    sub_171: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_337, sum_191);  mul_337 = sum_191 = None
    sub_172: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_171, mul_339);  sub_171 = mul_339 = None
    mul_340: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_172);  div_57 = sub_172 = None
    mul_341: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1003, mul_112);  mul_112 = None
    sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1]);  mul_341 = None
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1003, [0, 1]);  view_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_270: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_269, mul_340);  add_269 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1004: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_270, [128, 1024])
    mm_127: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1004, permute_733);  permute_733 = None
    permute_734: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1004, [1, 0])
    mm_128: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_734, view_469);  permute_734 = view_469 = None
    permute_735: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_195: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
    view_1005: "f32[1024]" = torch.ops.aten.reshape.default(sum_195, [1024]);  sum_195 = None
    permute_736: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    view_1006: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_127, [1, 128, 1024]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1007: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1006, [1, 128, 16, 64]);  view_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_737: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1007, [0, 2, 1, 3]);  view_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1008: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_737, [16, 128, 64]);  permute_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_120: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_738, view_1008);  permute_738 = None
    bmm_121: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1008, permute_739);  view_1008 = permute_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_342: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_121, div_23);  bmm_121 = None
    sum_196: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [-1], True)
    mul_343: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_23, sum_196);  div_23 = sum_196 = None
    sub_173: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_122: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_740, sub_173);  permute_740 = None
    bmm_123: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_173, permute_741);  sub_173 = permute_741 = None
    permute_742: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_122, [0, 2, 1]);  bmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1009: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_120, [1, 16, 128, 64]);  bmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_271: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_26, view_1009);  tangents_26 = view_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1010: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_742, [1, 16, 128, 64]);  permute_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_272: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_1010);  tangents_25 = view_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1011: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_123, [1, 16, 128, 64]);  bmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_743: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1011, [0, 2, 1, 3]);  view_1011 = None
    clone_302: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
    view_1012: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_302, [1, 128, 1024]);  clone_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_744: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_271, [0, 2, 1, 3]);  add_271 = None
    clone_303: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_744, memory_format = torch.contiguous_format);  permute_744 = None
    view_1013: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_303, [1, 128, 1024]);  clone_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1014: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1013, [128, 1024]);  view_1013 = None
    mm_129: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1014, permute_745);  permute_745 = None
    permute_746: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_130: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_746, view_267);  permute_746 = None
    permute_747: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[1024]" = torch.ops.aten.reshape.default(sum_197, [1024]);  sum_197 = None
    permute_748: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_747, [1, 0]);  permute_747 = None
    view_1016: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_129, [1, 128, 1024]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_273: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_263, view_1016);  add_263 = view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_749: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_272, [0, 2, 1, 3]);  add_272 = None
    clone_304: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
    view_1017: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_304, [1, 128, 1024]);  clone_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1018: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1017, [128, 1024]);  view_1017 = None
    mm_131: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1018, permute_750);  permute_750 = None
    permute_751: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1018, [1, 0])
    mm_132: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_751, view_267);  permute_751 = None
    permute_752: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True);  view_1018 = None
    view_1019: "f32[1024]" = torch.ops.aten.reshape.default(sum_198, [1024]);  sum_198 = None
    permute_753: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_752, [1, 0]);  permute_752 = None
    view_1020: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 1024]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_274: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_273, view_1020);  add_273 = view_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_344: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1012, 0.125);  view_1012 = None
    view_1021: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_344, [128, 1024]);  mul_344 = None
    mm_133: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1021, permute_754);  permute_754 = None
    permute_755: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1021, [1, 0])
    mm_134: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_755, view_455);  permute_755 = view_455 = None
    permute_756: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    sum_199: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1021, [0], True);  view_1021 = None
    view_1022: "f32[1024]" = torch.ops.aten.reshape.default(sum_199, [1024]);  sum_199 = None
    permute_757: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
    view_1023: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 1024]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_346: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1023, primals_337);  primals_337 = None
    mul_347: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_346, 1024)
    sum_200: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
    mul_348: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_346, mul_109);  mul_346 = None
    sum_201: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
    mul_349: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, sum_201);  sum_201 = None
    sub_175: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_347, sum_200);  mul_347 = sum_200 = None
    sub_176: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_175, mul_349);  sub_175 = mul_349 = None
    mul_350: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_176);  div_58 = sub_176 = None
    mul_351: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1023, mul_109);  mul_109 = None
    sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
    sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0, 1]);  view_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_275: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_270, mul_350);  add_270 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1024: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_275, [128, 1024])
    mm_135: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1024, permute_758);  permute_758 = None
    permute_759: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1024, [1, 0])
    mm_136: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_759, view_453);  permute_759 = view_453 = None
    permute_760: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_204: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1024, [0], True);  view_1024 = None
    view_1025: "f32[1024]" = torch.ops.aten.reshape.default(sum_204, [1024]);  sum_204 = None
    permute_761: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_760, [1, 0]);  permute_760 = None
    view_1026: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 1024]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1027: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1026, [1, 128, 16, 64]);  view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_762: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1027, [0, 2, 1, 3]);  view_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1028: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_762, [16, 128, 64]);  permute_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_124: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_763, view_1028);  permute_763 = None
    bmm_125: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1028, permute_764);  view_1028 = permute_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_352: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_125, alias_86);  bmm_125 = None
    sum_205: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
    mul_353: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_86, sum_205);  alias_86 = sum_205 = None
    sub_177: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1029: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_177, [1, 16, 128, 128]);  sub_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1030: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1029, [16, 128, 128]);  view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_126: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_765, view_1030);  permute_765 = None
    bmm_127: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1030, permute_766);  view_1030 = permute_766 = None
    permute_767: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_126, [0, 2, 1]);  bmm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1031: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_124, [1, 16, 128, 64]);  bmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_276: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_24, view_1031);  tangents_24 = view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1032: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_767, [1, 16, 128, 64]);  permute_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_277: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_1032);  tangents_23 = view_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1033: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_127, [1, 16, 128, 64]);  bmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_768: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1033, [0, 2, 1, 3]);  view_1033 = None
    clone_305: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_768, memory_format = torch.contiguous_format);  permute_768 = None
    view_1034: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_305, [1, 128, 1024]);  clone_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_769: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_276, [0, 2, 1, 3]);  add_276 = None
    clone_306: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_769, memory_format = torch.contiguous_format);  permute_769 = None
    view_1035: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_306, [1, 128, 1024]);  clone_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1036: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1035, [128, 1024]);  view_1035 = None
    mm_137: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1036, permute_770);  permute_770 = None
    permute_771: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1036, [1, 0])
    mm_138: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_771, view_437);  permute_771 = None
    permute_772: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_206: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1036, [0], True);  view_1036 = None
    view_1037: "f32[1024]" = torch.ops.aten.reshape.default(sum_206, [1024]);  sum_206 = None
    permute_773: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_772, [1, 0]);  permute_772 = None
    view_1038: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 1024]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_774: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_277, [0, 2, 1, 3]);  add_277 = None
    clone_307: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_774, memory_format = torch.contiguous_format);  permute_774 = None
    view_1039: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_307, [1, 128, 1024]);  clone_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1040: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1039, [128, 1024]);  view_1039 = None
    mm_139: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1040, permute_775);  permute_775 = None
    permute_776: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1040, [1, 0])
    mm_140: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_776, view_437);  permute_776 = None
    permute_777: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_207: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True);  view_1040 = None
    view_1041: "f32[1024]" = torch.ops.aten.reshape.default(sum_207, [1024]);  sum_207 = None
    permute_778: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_1042: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_139, [1, 128, 1024]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_278: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1038, view_1042);  view_1038 = view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_354: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1034, 0.125);  view_1034 = None
    view_1043: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_354, [128, 1024]);  mul_354 = None
    mm_141: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1043, permute_779);  permute_779 = None
    permute_780: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_142: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_780, view_437);  permute_780 = view_437 = None
    permute_781: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    sum_208: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[1024]" = torch.ops.aten.reshape.default(sum_208, [1024]);  sum_208 = None
    permute_782: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_1045: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [1, 128, 1024]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_279: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_278, view_1045);  add_278 = view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_356: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_279, primals_327);  primals_327 = None
    mul_357: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_356, 1024)
    sum_209: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
    mul_358: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_356, mul_106);  mul_356 = None
    sum_210: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
    mul_359: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_106, sum_210);  sum_210 = None
    sub_179: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_357, sum_209);  mul_357 = sum_209 = None
    sub_180: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_179, mul_359);  sub_179 = mul_359 = None
    mul_360: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_59, sub_180);  div_59 = sub_180 = None
    mul_361: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_279, mul_106);  mul_106 = None
    sum_211: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
    sum_212: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_279, [0, 1]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_280: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_275, mul_360);  add_275 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_1046: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_280, [128, 1024])
    mm_143: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1046, permute_783);  permute_783 = None
    permute_784: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1046, [1, 0])
    mm_144: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_784, view_435);  permute_784 = view_435 = None
    permute_785: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1046, [0], True);  view_1046 = None
    view_1047: "f32[1024]" = torch.ops.aten.reshape.default(sum_213, [1024]);  sum_213 = None
    permute_786: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_785, [1, 0]);  permute_785 = None
    view_1048: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_12: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_7, full_default_1, view_1048);  le_7 = view_1048 = None
    view_1049: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_12, [128, 4096]);  where_12 = None
    mm_145: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1049, permute_787);  permute_787 = None
    permute_788: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1049, [1, 0])
    mm_146: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_788, view_433);  permute_788 = view_433 = None
    permute_789: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    sum_214: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1049, [0], True);  view_1049 = None
    view_1050: "f32[4096]" = torch.ops.aten.reshape.default(sum_214, [4096]);  sum_214 = None
    permute_790: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_789, [1, 0]);  permute_789 = None
    view_1051: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_145, [1, 128, 1024]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_363: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1051, primals_321);  primals_321 = None
    mul_364: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_363, 1024)
    sum_215: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
    mul_365: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_363, mul_104);  mul_363 = None
    sum_216: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    mul_366: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_104, sum_216);  sum_216 = None
    sub_182: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_364, sum_215);  mul_364 = sum_215 = None
    sub_183: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_182, mul_366);  sub_182 = mul_366 = None
    mul_367: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_183);  div_60 = sub_183 = None
    mul_368: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1051, mul_104);  mul_104 = None
    sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
    sum_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0, 1]);  view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_281: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_280, mul_367);  add_280 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1052: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_281, [128, 1024])
    mm_147: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1052, permute_791);  permute_791 = None
    permute_792: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1052, [1, 0])
    mm_148: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_792, view_431);  permute_792 = view_431 = None
    permute_793: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_219: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1052, [0], True);  view_1052 = None
    view_1053: "f32[1024]" = torch.ops.aten.reshape.default(sum_219, [1024]);  sum_219 = None
    permute_794: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_793, [1, 0]);  permute_793 = None
    view_1054: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_147, [1, 128, 1024]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1055: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1054, [1, 128, 16, 64]);  view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_795: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1055, [0, 2, 1, 3]);  view_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1056: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_795, [16, 128, 64]);  permute_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_128: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_796, view_1056);  permute_796 = None
    bmm_129: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1056, permute_797);  view_1056 = permute_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_369: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_129, div_21);  bmm_129 = None
    sum_220: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [-1], True)
    mul_370: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_21, sum_220);  div_21 = sum_220 = None
    sub_184: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_130: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_798, sub_184);  permute_798 = None
    bmm_131: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_184, permute_799);  sub_184 = permute_799 = None
    permute_800: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_130, [0, 2, 1]);  bmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1057: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_128, [1, 16, 128, 64]);  bmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_282: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_22, view_1057);  tangents_22 = view_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1058: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_800, [1, 16, 128, 64]);  permute_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_283: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_1058);  tangents_21 = view_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1059: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_131, [1, 16, 128, 64]);  bmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_801: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1059, [0, 2, 1, 3]);  view_1059 = None
    clone_308: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_801, memory_format = torch.contiguous_format);  permute_801 = None
    view_1060: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_308, [1, 128, 1024]);  clone_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_802: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_282, [0, 2, 1, 3]);  add_282 = None
    clone_309: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_802, memory_format = torch.contiguous_format);  permute_802 = None
    view_1061: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_309, [1, 128, 1024]);  clone_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1062: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1061, [128, 1024]);  view_1061 = None
    mm_149: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1062, permute_803);  permute_803 = None
    permute_804: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_150: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_804, view_267);  permute_804 = None
    permute_805: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_221: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
    view_1063: "f32[1024]" = torch.ops.aten.reshape.default(sum_221, [1024]);  sum_221 = None
    permute_806: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    view_1064: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_149, [1, 128, 1024]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_284: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_274, view_1064);  add_274 = view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_807: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_283, [0, 2, 1, 3]);  add_283 = None
    clone_310: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_807, memory_format = torch.contiguous_format);  permute_807 = None
    view_1065: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_310, [1, 128, 1024]);  clone_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1066: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1065, [128, 1024]);  view_1065 = None
    mm_151: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1066, permute_808);  permute_808 = None
    permute_809: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1066, [1, 0])
    mm_152: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_809, view_267);  permute_809 = None
    permute_810: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_222: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
    view_1067: "f32[1024]" = torch.ops.aten.reshape.default(sum_222, [1024]);  sum_222 = None
    permute_811: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    view_1068: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_151, [1, 128, 1024]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_285: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_284, view_1068);  add_284 = view_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_371: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1060, 0.125);  view_1060 = None
    view_1069: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_371, [128, 1024]);  mul_371 = None
    mm_153: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1069, permute_812);  permute_812 = None
    permute_813: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1069, [1, 0])
    mm_154: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_813, view_417);  permute_813 = view_417 = None
    permute_814: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_223: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1069, [0], True);  view_1069 = None
    view_1070: "f32[1024]" = torch.ops.aten.reshape.default(sum_223, [1024]);  sum_223 = None
    permute_815: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_1071: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_153, [1, 128, 1024]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_373: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1071, primals_311);  primals_311 = None
    mul_374: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_373, 1024)
    sum_224: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True)
    mul_375: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_373, mul_101);  mul_373 = None
    sum_225: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    mul_376: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_101, sum_225);  sum_225 = None
    sub_186: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_374, sum_224);  mul_374 = sum_224 = None
    sub_187: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_186, mul_376);  sub_186 = mul_376 = None
    mul_377: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_187);  div_61 = sub_187 = None
    mul_378: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1071, mul_101);  mul_101 = None
    sum_226: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1]);  mul_378 = None
    sum_227: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1071, [0, 1]);  view_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_286: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_281, mul_377);  add_281 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1072: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_286, [128, 1024])
    mm_155: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1072, permute_816);  permute_816 = None
    permute_817: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1072, [1, 0])
    mm_156: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_817, view_415);  permute_817 = view_415 = None
    permute_818: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1072, [0], True);  view_1072 = None
    view_1073: "f32[1024]" = torch.ops.aten.reshape.default(sum_228, [1024]);  sum_228 = None
    permute_819: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_818, [1, 0]);  permute_818 = None
    view_1074: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_155, [1, 128, 1024]);  mm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1075: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1074, [1, 128, 16, 64]);  view_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_820: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1075, [0, 2, 1, 3]);  view_1075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1076: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_820, [16, 128, 64]);  permute_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_132: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_821, view_1076);  permute_821 = None
    bmm_133: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1076, permute_822);  view_1076 = permute_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_379: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_133, alias_89);  bmm_133 = None
    sum_229: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_229);  alias_89 = sum_229 = None
    sub_188: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1077: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_188, [1, 16, 128, 128]);  sub_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1078: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1077, [16, 128, 128]);  view_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_134: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_823, view_1078);  permute_823 = None
    bmm_135: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1078, permute_824);  view_1078 = permute_824 = None
    permute_825: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_134, [0, 2, 1]);  bmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1079: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_132, [1, 16, 128, 64]);  bmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_287: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_20, view_1079);  tangents_20 = view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1080: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_825, [1, 16, 128, 64]);  permute_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_288: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_1080);  tangents_19 = view_1080 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1081: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_135, [1, 16, 128, 64]);  bmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_826: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1081, [0, 2, 1, 3]);  view_1081 = None
    clone_311: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_826, memory_format = torch.contiguous_format);  permute_826 = None
    view_1082: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_311, [1, 128, 1024]);  clone_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_827: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_287, [0, 2, 1, 3]);  add_287 = None
    clone_312: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_827, memory_format = torch.contiguous_format);  permute_827 = None
    view_1083: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_312, [1, 128, 1024]);  clone_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1084: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1083, [128, 1024]);  view_1083 = None
    mm_157: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1084, permute_828);  permute_828 = None
    permute_829: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1084, [1, 0])
    mm_158: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_829, view_399);  permute_829 = None
    permute_830: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    sum_230: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1084, [0], True);  view_1084 = None
    view_1085: "f32[1024]" = torch.ops.aten.reshape.default(sum_230, [1024]);  sum_230 = None
    permute_831: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_830, [1, 0]);  permute_830 = None
    view_1086: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_157, [1, 128, 1024]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_832: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_288, [0, 2, 1, 3]);  add_288 = None
    clone_313: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_832, memory_format = torch.contiguous_format);  permute_832 = None
    view_1087: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_313, [1, 128, 1024]);  clone_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1088: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1087, [128, 1024]);  view_1087 = None
    mm_159: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1088, permute_833);  permute_833 = None
    permute_834: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1088, [1, 0])
    mm_160: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_834, view_399);  permute_834 = None
    permute_835: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_231: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1088, [0], True);  view_1088 = None
    view_1089: "f32[1024]" = torch.ops.aten.reshape.default(sum_231, [1024]);  sum_231 = None
    permute_836: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_835, [1, 0]);  permute_835 = None
    view_1090: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_159, [1, 128, 1024]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_289: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1086, view_1090);  view_1086 = view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_381: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1082, 0.125);  view_1082 = None
    view_1091: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_381, [128, 1024]);  mul_381 = None
    mm_161: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1091, permute_837);  permute_837 = None
    permute_838: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1091, [1, 0])
    mm_162: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_838, view_399);  permute_838 = view_399 = None
    permute_839: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_232: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1091, [0], True);  view_1091 = None
    view_1092: "f32[1024]" = torch.ops.aten.reshape.default(sum_232, [1024]);  sum_232 = None
    permute_840: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_839, [1, 0]);  permute_839 = None
    view_1093: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_161, [1, 128, 1024]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_290: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_289, view_1093);  add_289 = view_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_383: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_301);  primals_301 = None
    mul_384: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_383, 1024)
    sum_233: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_98);  mul_383 = None
    sum_234: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, sum_234);  sum_234 = None
    sub_190: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_384, sum_233);  mul_384 = sum_233 = None
    sub_191: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_190, mul_386);  sub_190 = mul_386 = None
    mul_387: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_62, sub_191);  div_62 = sub_191 = None
    mul_388: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_98);  mul_98 = None
    sum_235: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_236: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_291: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_286, mul_387);  add_286 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_1094: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_291, [128, 1024])
    mm_163: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1094, permute_841);  permute_841 = None
    permute_842: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1094, [1, 0])
    mm_164: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_842, view_397);  permute_842 = view_397 = None
    permute_843: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_237: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1094, [0], True);  view_1094 = None
    view_1095: "f32[1024]" = torch.ops.aten.reshape.default(sum_237, [1024]);  sum_237 = None
    permute_844: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    view_1096: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_163, [1, 128, 4096]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_13: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_8, full_default_1, view_1096);  le_8 = view_1096 = None
    view_1097: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_13, [128, 4096]);  where_13 = None
    mm_165: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1097, permute_845);  permute_845 = None
    permute_846: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1097, [1, 0])
    mm_166: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_846, view_395);  permute_846 = view_395 = None
    permute_847: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    sum_238: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1097, [0], True);  view_1097 = None
    view_1098: "f32[4096]" = torch.ops.aten.reshape.default(sum_238, [4096]);  sum_238 = None
    permute_848: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    view_1099: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_165, [1, 128, 1024]);  mm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_390: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1099, primals_295);  primals_295 = None
    mul_391: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_390, 1024)
    sum_239: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
    mul_392: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_390, mul_96);  mul_390 = None
    sum_240: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
    mul_393: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_96, sum_240);  sum_240 = None
    sub_193: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_391, sum_239);  mul_391 = sum_239 = None
    sub_194: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_193, mul_393);  sub_193 = mul_393 = None
    mul_394: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_194);  div_63 = sub_194 = None
    mul_395: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1099, mul_96);  mul_96 = None
    sum_241: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
    sum_242: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1099, [0, 1]);  view_1099 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_292: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_291, mul_394);  add_291 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1100: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_292, [128, 1024])
    mm_167: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1100, permute_849);  permute_849 = None
    permute_850: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1100, [1, 0])
    mm_168: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_850, view_393);  permute_850 = view_393 = None
    permute_851: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1100, [0], True);  view_1100 = None
    view_1101: "f32[1024]" = torch.ops.aten.reshape.default(sum_243, [1024]);  sum_243 = None
    permute_852: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_851, [1, 0]);  permute_851 = None
    view_1102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_167, [1, 128, 1024]);  mm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1103: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1102, [1, 128, 16, 64]);  view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_853: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1103, [0, 2, 1, 3]);  view_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1104: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_853, [16, 128, 64]);  permute_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_136: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_854, view_1104);  permute_854 = None
    bmm_137: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1104, permute_855);  view_1104 = permute_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_396: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_137, div_19);  bmm_137 = None
    sum_244: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [-1], True)
    mul_397: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_19, sum_244);  div_19 = sum_244 = None
    sub_195: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_138: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_856, sub_195);  permute_856 = None
    bmm_139: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_195, permute_857);  sub_195 = permute_857 = None
    permute_858: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_138, [0, 2, 1]);  bmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1105: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_136, [1, 16, 128, 64]);  bmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_293: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_18, view_1105);  tangents_18 = view_1105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1106: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_858, [1, 16, 128, 64]);  permute_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_294: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_1106);  tangents_17 = view_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1107: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_139, [1, 16, 128, 64]);  bmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_859: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1107, [0, 2, 1, 3]);  view_1107 = None
    clone_314: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_859, memory_format = torch.contiguous_format);  permute_859 = None
    view_1108: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_314, [1, 128, 1024]);  clone_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_860: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_293, [0, 2, 1, 3]);  add_293 = None
    clone_315: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_860, memory_format = torch.contiguous_format);  permute_860 = None
    view_1109: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_315, [1, 128, 1024]);  clone_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1110: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1109, [128, 1024]);  view_1109 = None
    mm_169: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1110, permute_861);  permute_861 = None
    permute_862: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1110, [1, 0])
    mm_170: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_862, view_267);  permute_862 = None
    permute_863: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    sum_245: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1110, [0], True);  view_1110 = None
    view_1111: "f32[1024]" = torch.ops.aten.reshape.default(sum_245, [1024]);  sum_245 = None
    permute_864: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
    view_1112: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_169, [1, 128, 1024]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_295: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_285, view_1112);  add_285 = view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_865: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_294, [0, 2, 1, 3]);  add_294 = None
    clone_316: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_865, memory_format = torch.contiguous_format);  permute_865 = None
    view_1113: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_316, [1, 128, 1024]);  clone_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1114: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1113, [128, 1024]);  view_1113 = None
    mm_171: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1114, permute_866);  permute_866 = None
    permute_867: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1114, [1, 0])
    mm_172: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_867, view_267);  permute_867 = None
    permute_868: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_246: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1114, [0], True);  view_1114 = None
    view_1115: "f32[1024]" = torch.ops.aten.reshape.default(sum_246, [1024]);  sum_246 = None
    permute_869: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_868, [1, 0]);  permute_868 = None
    view_1116: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_171, [1, 128, 1024]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_296: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_295, view_1116);  add_295 = view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_398: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1108, 0.125);  view_1108 = None
    view_1117: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_398, [128, 1024]);  mul_398 = None
    mm_173: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1117, permute_870);  permute_870 = None
    permute_871: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1117, [1, 0])
    mm_174: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_871, view_379);  permute_871 = view_379 = None
    permute_872: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_247: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1117, [0], True);  view_1117 = None
    view_1118: "f32[1024]" = torch.ops.aten.reshape.default(sum_247, [1024]);  sum_247 = None
    permute_873: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_872, [1, 0]);  permute_872 = None
    view_1119: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_173, [1, 128, 1024]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_400: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1119, primals_285);  primals_285 = None
    mul_401: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_400, 1024)
    sum_248: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True)
    mul_402: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_400, mul_93);  mul_400 = None
    sum_249: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
    mul_403: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_93, sum_249);  sum_249 = None
    sub_197: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_401, sum_248);  mul_401 = sum_248 = None
    sub_198: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_197, mul_403);  sub_197 = mul_403 = None
    mul_404: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_198);  div_64 = sub_198 = None
    mul_405: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1119, mul_93);  mul_93 = None
    sum_250: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1]);  mul_405 = None
    sum_251: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1119, [0, 1]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_297: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_292, mul_404);  add_292 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1120: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_297, [128, 1024])
    mm_175: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1120, permute_874);  permute_874 = None
    permute_875: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1120, [1, 0])
    mm_176: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_875, view_377);  permute_875 = view_377 = None
    permute_876: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_252: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1120, [0], True);  view_1120 = None
    view_1121: "f32[1024]" = torch.ops.aten.reshape.default(sum_252, [1024]);  sum_252 = None
    permute_877: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    view_1122: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_175, [1, 128, 1024]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1123: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1122, [1, 128, 16, 64]);  view_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_878: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1123, [0, 2, 1, 3]);  view_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1124: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_878, [16, 128, 64]);  permute_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_140: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_879, view_1124);  permute_879 = None
    bmm_141: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1124, permute_880);  view_1124 = permute_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_406: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_141, alias_92);  bmm_141 = None
    sum_253: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [-1], True)
    mul_407: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_92, sum_253);  alias_92 = sum_253 = None
    sub_199: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1125: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_199, [1, 16, 128, 128]);  sub_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1126: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1125, [16, 128, 128]);  view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_142: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_881, view_1126);  permute_881 = None
    bmm_143: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1126, permute_882);  view_1126 = permute_882 = None
    permute_883: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_142, [0, 2, 1]);  bmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1127: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_140, [1, 16, 128, 64]);  bmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_298: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_16, view_1127);  tangents_16 = view_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1128: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_883, [1, 16, 128, 64]);  permute_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_299: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_1128);  tangents_15 = view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1129: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_143, [1, 16, 128, 64]);  bmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_884: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1129, [0, 2, 1, 3]);  view_1129 = None
    clone_317: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_884, memory_format = torch.contiguous_format);  permute_884 = None
    view_1130: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_317, [1, 128, 1024]);  clone_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_885: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_298, [0, 2, 1, 3]);  add_298 = None
    clone_318: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_885, memory_format = torch.contiguous_format);  permute_885 = None
    view_1131: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_318, [1, 128, 1024]);  clone_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1132: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1131, [128, 1024]);  view_1131 = None
    mm_177: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1132, permute_886);  permute_886 = None
    permute_887: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
    mm_178: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_887, view_361);  permute_887 = None
    permute_888: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    sum_254: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
    view_1133: "f32[1024]" = torch.ops.aten.reshape.default(sum_254, [1024]);  sum_254 = None
    permute_889: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
    view_1134: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_177, [1, 128, 1024]);  mm_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_890: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_299, [0, 2, 1, 3]);  add_299 = None
    clone_319: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
    view_1135: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_319, [1, 128, 1024]);  clone_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1136: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1135, [128, 1024]);  view_1135 = None
    mm_179: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1136, permute_891);  permute_891 = None
    permute_892: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1136, [1, 0])
    mm_180: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_892, view_361);  permute_892 = None
    permute_893: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1136, [0], True);  view_1136 = None
    view_1137: "f32[1024]" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
    permute_894: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_893, [1, 0]);  permute_893 = None
    view_1138: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_179, [1, 128, 1024]);  mm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_300: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1134, view_1138);  view_1134 = view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_408: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1130, 0.125);  view_1130 = None
    view_1139: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_408, [128, 1024]);  mul_408 = None
    mm_181: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1139, permute_895);  permute_895 = None
    permute_896: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1139, [1, 0])
    mm_182: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_896, view_361);  permute_896 = view_361 = None
    permute_897: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    sum_256: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1139, [0], True);  view_1139 = None
    view_1140: "f32[1024]" = torch.ops.aten.reshape.default(sum_256, [1024]);  sum_256 = None
    permute_898: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_897, [1, 0]);  permute_897 = None
    view_1141: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_181, [1, 128, 1024]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_301: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_300, view_1141);  add_300 = view_1141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_410: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_301, primals_275);  primals_275 = None
    mul_411: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_410, 1024)
    sum_257: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [2], True)
    mul_412: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_410, mul_90);  mul_410 = None
    sum_258: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True);  mul_412 = None
    mul_413: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_90, sum_258);  sum_258 = None
    sub_201: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_411, sum_257);  mul_411 = sum_257 = None
    sub_202: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_201, mul_413);  sub_201 = mul_413 = None
    mul_414: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_65, sub_202);  div_65 = sub_202 = None
    mul_415: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_301, mul_90);  mul_90 = None
    sum_259: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1]);  mul_415 = None
    sum_260: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_301, [0, 1]);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_302: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_297, mul_414);  add_297 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_1142: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_302, [128, 1024])
    mm_183: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1142, permute_899);  permute_899 = None
    permute_900: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1142, [1, 0])
    mm_184: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_900, view_359);  permute_900 = view_359 = None
    permute_901: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_261: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1142, [0], True);  view_1142 = None
    view_1143: "f32[1024]" = torch.ops.aten.reshape.default(sum_261, [1024]);  sum_261 = None
    permute_902: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_901, [1, 0]);  permute_901 = None
    view_1144: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_14: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_9, full_default_1, view_1144);  le_9 = view_1144 = None
    view_1145: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_14, [128, 4096]);  where_14 = None
    mm_185: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1145, permute_903);  permute_903 = None
    permute_904: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1145, [1, 0])
    mm_186: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_904, view_357);  permute_904 = view_357 = None
    permute_905: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_262: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1145, [0], True);  view_1145 = None
    view_1146: "f32[4096]" = torch.ops.aten.reshape.default(sum_262, [4096]);  sum_262 = None
    permute_906: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_905, [1, 0]);  permute_905 = None
    view_1147: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_185, [1, 128, 1024]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_417: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1147, primals_269);  primals_269 = None
    mul_418: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_417, 1024)
    sum_263: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True)
    mul_419: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_417, mul_88);  mul_417 = None
    sum_264: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True);  mul_419 = None
    mul_420: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_88, sum_264);  sum_264 = None
    sub_204: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_418, sum_263);  mul_418 = sum_263 = None
    sub_205: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_204, mul_420);  sub_204 = mul_420 = None
    mul_421: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_205);  div_66 = sub_205 = None
    mul_422: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1147, mul_88);  mul_88 = None
    sum_265: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 1]);  mul_422 = None
    sum_266: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1147, [0, 1]);  view_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_303: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_302, mul_421);  add_302 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1148: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_303, [128, 1024])
    mm_187: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1148, permute_907);  permute_907 = None
    permute_908: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1148, [1, 0])
    mm_188: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_908, view_355);  permute_908 = view_355 = None
    permute_909: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_267: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1148, [0], True);  view_1148 = None
    view_1149: "f32[1024]" = torch.ops.aten.reshape.default(sum_267, [1024]);  sum_267 = None
    permute_910: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_909, [1, 0]);  permute_909 = None
    view_1150: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_187, [1, 128, 1024]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1151: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1150, [1, 128, 16, 64]);  view_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_911: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1151, [0, 2, 1, 3]);  view_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1152: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_911, [16, 128, 64]);  permute_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_144: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_912, view_1152);  permute_912 = None
    bmm_145: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1152, permute_913);  view_1152 = permute_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_423: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_145, div_17);  bmm_145 = None
    sum_268: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [-1], True)
    mul_424: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_17, sum_268);  div_17 = sum_268 = None
    sub_206: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_423, mul_424);  mul_423 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_146: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_914, sub_206);  permute_914 = None
    bmm_147: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_206, permute_915);  sub_206 = permute_915 = None
    permute_916: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_146, [0, 2, 1]);  bmm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1153: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_144, [1, 16, 128, 64]);  bmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_304: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_14, view_1153);  tangents_14 = view_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1154: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_916, [1, 16, 128, 64]);  permute_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_305: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_1154);  tangents_13 = view_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1155: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_147, [1, 16, 128, 64]);  bmm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_917: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1155, [0, 2, 1, 3]);  view_1155 = None
    clone_320: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_917, memory_format = torch.contiguous_format);  permute_917 = None
    view_1156: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_320, [1, 128, 1024]);  clone_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_918: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_304, [0, 2, 1, 3]);  add_304 = None
    clone_321: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_918, memory_format = torch.contiguous_format);  permute_918 = None
    view_1157: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_321, [1, 128, 1024]);  clone_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1158: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1157, [128, 1024]);  view_1157 = None
    mm_189: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1158, permute_919);  permute_919 = None
    permute_920: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1158, [1, 0])
    mm_190: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_920, view_267);  permute_920 = None
    permute_921: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    sum_269: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
    view_1159: "f32[1024]" = torch.ops.aten.reshape.default(sum_269, [1024]);  sum_269 = None
    permute_922: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_921, [1, 0]);  permute_921 = None
    view_1160: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_189, [1, 128, 1024]);  mm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_306: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_296, view_1160);  add_296 = view_1160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_923: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_305, [0, 2, 1, 3]);  add_305 = None
    clone_322: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
    view_1161: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_322, [1, 128, 1024]);  clone_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1162: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1161, [128, 1024]);  view_1161 = None
    mm_191: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1162, permute_924);  permute_924 = None
    permute_925: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1162, [1, 0])
    mm_192: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_925, view_267);  permute_925 = None
    permute_926: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    sum_270: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1162, [0], True);  view_1162 = None
    view_1163: "f32[1024]" = torch.ops.aten.reshape.default(sum_270, [1024]);  sum_270 = None
    permute_927: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_926, [1, 0]);  permute_926 = None
    view_1164: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_191, [1, 128, 1024]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_307: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_306, view_1164);  add_306 = view_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_425: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1156, 0.125);  view_1156 = None
    view_1165: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_425, [128, 1024]);  mul_425 = None
    mm_193: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1165, permute_928);  permute_928 = None
    permute_929: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1165, [1, 0])
    mm_194: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_929, view_341);  permute_929 = view_341 = None
    permute_930: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    sum_271: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1165, [0], True);  view_1165 = None
    view_1166: "f32[1024]" = torch.ops.aten.reshape.default(sum_271, [1024]);  sum_271 = None
    permute_931: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_930, [1, 0]);  permute_930 = None
    view_1167: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_193, [1, 128, 1024]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_427: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1167, primals_259);  primals_259 = None
    mul_428: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_427, 1024)
    sum_272: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_427, mul_85);  mul_427 = None
    sum_273: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_85, sum_273);  sum_273 = None
    sub_208: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_428, sum_272);  mul_428 = sum_272 = None
    sub_209: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_208, mul_430);  sub_208 = mul_430 = None
    mul_431: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_209);  div_67 = sub_209 = None
    mul_432: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1167, mul_85);  mul_85 = None
    sum_274: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_275: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1167, [0, 1]);  view_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_308: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_303, mul_431);  add_303 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1168: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_308, [128, 1024])
    mm_195: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1168, permute_932);  permute_932 = None
    permute_933: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1168, [1, 0])
    mm_196: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_933, view_339);  permute_933 = view_339 = None
    permute_934: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_276: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1168, [0], True);  view_1168 = None
    view_1169: "f32[1024]" = torch.ops.aten.reshape.default(sum_276, [1024]);  sum_276 = None
    permute_935: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_934, [1, 0]);  permute_934 = None
    view_1170: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_195, [1, 128, 1024]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1171: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1170, [1, 128, 16, 64]);  view_1170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_936: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1171, [0, 2, 1, 3]);  view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1172: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_936, [16, 128, 64]);  permute_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_148: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_937, view_1172);  permute_937 = None
    bmm_149: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1172, permute_938);  view_1172 = permute_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_433: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_149, alias_95);  bmm_149 = None
    sum_277: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [-1], True)
    mul_434: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_277);  alias_95 = sum_277 = None
    sub_210: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1173: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_210, [1, 16, 128, 128]);  sub_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1174: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1173, [16, 128, 128]);  view_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_150: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_939, view_1174);  permute_939 = None
    bmm_151: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1174, permute_940);  view_1174 = permute_940 = None
    permute_941: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_150, [0, 2, 1]);  bmm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1175: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_148, [1, 16, 128, 64]);  bmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_309: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_12, view_1175);  tangents_12 = view_1175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1176: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_941, [1, 16, 128, 64]);  permute_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_310: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_1176);  tangents_11 = view_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1177: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_151, [1, 16, 128, 64]);  bmm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_942: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1177, [0, 2, 1, 3]);  view_1177 = None
    clone_323: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_942, memory_format = torch.contiguous_format);  permute_942 = None
    view_1178: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_323, [1, 128, 1024]);  clone_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_943: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_309, [0, 2, 1, 3]);  add_309 = None
    clone_324: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_943, memory_format = torch.contiguous_format);  permute_943 = None
    view_1179: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_324, [1, 128, 1024]);  clone_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1180: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1179, [128, 1024]);  view_1179 = None
    mm_197: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1180, permute_944);  permute_944 = None
    permute_945: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1180, [1, 0])
    mm_198: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_945, view_323);  permute_945 = None
    permute_946: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_198, [1, 0]);  mm_198 = None
    sum_278: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1180, [0], True);  view_1180 = None
    view_1181: "f32[1024]" = torch.ops.aten.reshape.default(sum_278, [1024]);  sum_278 = None
    permute_947: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_946, [1, 0]);  permute_946 = None
    view_1182: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_197, [1, 128, 1024]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_948: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_310, [0, 2, 1, 3]);  add_310 = None
    clone_325: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
    view_1183: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_325, [1, 128, 1024]);  clone_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1184: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1183, [128, 1024]);  view_1183 = None
    mm_199: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1184, permute_949);  permute_949 = None
    permute_950: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1184, [1, 0])
    mm_200: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_950, view_323);  permute_950 = None
    permute_951: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_279: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True);  view_1184 = None
    view_1185: "f32[1024]" = torch.ops.aten.reshape.default(sum_279, [1024]);  sum_279 = None
    permute_952: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_951, [1, 0]);  permute_951 = None
    view_1186: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_199, [1, 128, 1024]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_311: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1182, view_1186);  view_1182 = view_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_435: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1178, 0.125);  view_1178 = None
    view_1187: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_435, [128, 1024]);  mul_435 = None
    mm_201: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1187, permute_953);  permute_953 = None
    permute_954: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1187, [1, 0])
    mm_202: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_954, view_323);  permute_954 = view_323 = None
    permute_955: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    sum_280: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1187, [0], True);  view_1187 = None
    view_1188: "f32[1024]" = torch.ops.aten.reshape.default(sum_280, [1024]);  sum_280 = None
    permute_956: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_955, [1, 0]);  permute_955 = None
    view_1189: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_201, [1, 128, 1024]);  mm_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_312: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_311, view_1189);  add_311 = view_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_437: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_312, primals_249);  primals_249 = None
    mul_438: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_437, 1024)
    sum_281: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
    mul_439: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_437, mul_82);  mul_437 = None
    sum_282: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
    mul_440: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_82, sum_282);  sum_282 = None
    sub_212: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_438, sum_281);  mul_438 = sum_281 = None
    sub_213: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_212, mul_440);  sub_212 = mul_440 = None
    mul_441: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_68, sub_213);  div_68 = sub_213 = None
    mul_442: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_312, mul_82);  mul_82 = None
    sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
    sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_312, [0, 1]);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_313: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_308, mul_441);  add_308 = mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_1190: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_313, [128, 1024])
    mm_203: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1190, permute_957);  permute_957 = None
    permute_958: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1190, [1, 0])
    mm_204: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_958, view_321);  permute_958 = view_321 = None
    permute_959: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_204, [1, 0]);  mm_204 = None
    sum_285: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
    view_1191: "f32[1024]" = torch.ops.aten.reshape.default(sum_285, [1024]);  sum_285 = None
    permute_960: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_959, [1, 0]);  permute_959 = None
    view_1192: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_15: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_10, full_default_1, view_1192);  le_10 = view_1192 = None
    view_1193: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_15, [128, 4096]);  where_15 = None
    mm_205: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1193, permute_961);  permute_961 = None
    permute_962: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1193, [1, 0])
    mm_206: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_962, view_319);  permute_962 = view_319 = None
    permute_963: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_206, [1, 0]);  mm_206 = None
    sum_286: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1193, [0], True);  view_1193 = None
    view_1194: "f32[4096]" = torch.ops.aten.reshape.default(sum_286, [4096]);  sum_286 = None
    permute_964: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_963, [1, 0]);  permute_963 = None
    view_1195: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_205, [1, 128, 1024]);  mm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_444: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1195, primals_243);  primals_243 = None
    mul_445: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_444, 1024)
    sum_287: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True)
    mul_446: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_444, mul_80);  mul_444 = None
    sum_288: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
    mul_447: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_80, sum_288);  sum_288 = None
    sub_215: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_445, sum_287);  mul_445 = sum_287 = None
    sub_216: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_215, mul_447);  sub_215 = mul_447 = None
    mul_448: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_216);  div_69 = sub_216 = None
    mul_449: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1195, mul_80);  mul_80 = None
    sum_289: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 1]);  mul_449 = None
    sum_290: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1195, [0, 1]);  view_1195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_314: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_313, mul_448);  add_313 = mul_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1196: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_314, [128, 1024])
    mm_207: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1196, permute_965);  permute_965 = None
    permute_966: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1196, [1, 0])
    mm_208: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_966, view_317);  permute_966 = view_317 = None
    permute_967: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    sum_291: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1196, [0], True);  view_1196 = None
    view_1197: "f32[1024]" = torch.ops.aten.reshape.default(sum_291, [1024]);  sum_291 = None
    permute_968: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_967, [1, 0]);  permute_967 = None
    view_1198: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_207, [1, 128, 1024]);  mm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1199: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1198, [1, 128, 16, 64]);  view_1198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_969: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1199, [0, 2, 1, 3]);  view_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1200: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_969, [16, 128, 64]);  permute_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_152: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_970, view_1200);  permute_970 = None
    bmm_153: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1200, permute_971);  view_1200 = permute_971 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_450: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_153, div_15);  bmm_153 = None
    sum_292: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [-1], True)
    mul_451: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_15, sum_292);  div_15 = sum_292 = None
    sub_217: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_154: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_972, sub_217);  permute_972 = None
    bmm_155: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_217, permute_973);  sub_217 = permute_973 = None
    permute_974: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_154, [0, 2, 1]);  bmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1201: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_152, [1, 16, 128, 64]);  bmm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_315: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_10, view_1201);  tangents_10 = view_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1202: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_974, [1, 16, 128, 64]);  permute_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_316: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_1202);  tangents_9 = view_1202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1203: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_155, [1, 16, 128, 64]);  bmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_975: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None
    clone_326: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_975, memory_format = torch.contiguous_format);  permute_975 = None
    view_1204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_326, [1, 128, 1024]);  clone_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_976: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_315, [0, 2, 1, 3]);  add_315 = None
    clone_327: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_976, memory_format = torch.contiguous_format);  permute_976 = None
    view_1205: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_327, [1, 128, 1024]);  clone_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1206: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1205, [128, 1024]);  view_1205 = None
    mm_209: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1206, permute_977);  permute_977 = None
    permute_978: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1206, [1, 0])
    mm_210: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_978, view_267);  permute_978 = None
    permute_979: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_210, [1, 0]);  mm_210 = None
    sum_293: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1206, [0], True);  view_1206 = None
    view_1207: "f32[1024]" = torch.ops.aten.reshape.default(sum_293, [1024]);  sum_293 = None
    permute_980: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_979, [1, 0]);  permute_979 = None
    view_1208: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_209, [1, 128, 1024]);  mm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_317: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_307, view_1208);  add_307 = view_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_981: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_316, [0, 2, 1, 3]);  add_316 = None
    clone_328: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_981, memory_format = torch.contiguous_format);  permute_981 = None
    view_1209: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_328, [1, 128, 1024]);  clone_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1210: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1209, [128, 1024]);  view_1209 = None
    mm_211: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1210, permute_982);  permute_982 = None
    permute_983: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1210, [1, 0])
    mm_212: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_983, view_267);  permute_983 = None
    permute_984: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_212, [1, 0]);  mm_212 = None
    sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
    view_1211: "f32[1024]" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
    permute_985: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_984, [1, 0]);  permute_984 = None
    view_1212: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_211, [1, 128, 1024]);  mm_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_318: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_317, view_1212);  add_317 = view_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_452: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1204, 0.125);  view_1204 = None
    view_1213: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_452, [128, 1024]);  mul_452 = None
    mm_213: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1213, permute_986);  permute_986 = None
    permute_987: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1213, [1, 0])
    mm_214: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_987, view_303);  permute_987 = view_303 = None
    permute_988: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_214, [1, 0]);  mm_214 = None
    sum_295: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
    view_1214: "f32[1024]" = torch.ops.aten.reshape.default(sum_295, [1024]);  sum_295 = None
    permute_989: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_988, [1, 0]);  permute_988 = None
    view_1215: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_213, [1, 128, 1024]);  mm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_454: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1215, primals_233);  primals_233 = None
    mul_455: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_454, 1024)
    sum_296: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_454, mul_77);  mul_454 = None
    sum_297: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_77, sum_297);  sum_297 = None
    sub_219: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_455, sum_296);  mul_455 = sum_296 = None
    sub_220: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_219, mul_457);  sub_219 = mul_457 = None
    mul_458: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_220);  div_70 = sub_220 = None
    mul_459: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1215, mul_77);  mul_77 = None
    sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1215, [0, 1]);  view_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_319: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_314, mul_458);  add_314 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1216: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_319, [128, 1024])
    mm_215: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1216, permute_990);  permute_990 = None
    permute_991: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1216, [1, 0])
    mm_216: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_991, view_301);  permute_991 = view_301 = None
    permute_992: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_216, [1, 0]);  mm_216 = None
    sum_300: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1216, [0], True);  view_1216 = None
    view_1217: "f32[1024]" = torch.ops.aten.reshape.default(sum_300, [1024]);  sum_300 = None
    permute_993: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_992, [1, 0]);  permute_992 = None
    view_1218: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_215, [1, 128, 1024]);  mm_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1219: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1218, [1, 128, 16, 64]);  view_1218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_994: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1219, [0, 2, 1, 3]);  view_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1220: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_994, [16, 128, 64]);  permute_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_156: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_995, view_1220);  permute_995 = None
    bmm_157: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1220, permute_996);  view_1220 = permute_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_460: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_157, alias_98);  bmm_157 = None
    sum_301: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [-1], True)
    mul_461: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_98, sum_301);  alias_98 = sum_301 = None
    sub_221: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1221: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_221, [1, 16, 128, 128]);  sub_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1222: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1221, [16, 128, 128]);  view_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_158: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_997, view_1222);  permute_997 = None
    bmm_159: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1222, permute_998);  view_1222 = permute_998 = None
    permute_999: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_158, [0, 2, 1]);  bmm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1223: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_156, [1, 16, 128, 64]);  bmm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_320: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_8, view_1223);  tangents_8 = view_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1224: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_999, [1, 16, 128, 64]);  permute_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_321: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_1224);  tangents_7 = view_1224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1225: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_159, [1, 16, 128, 64]);  bmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1000: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1225, [0, 2, 1, 3]);  view_1225 = None
    clone_329: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1000, memory_format = torch.contiguous_format);  permute_1000 = None
    view_1226: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_329, [1, 128, 1024]);  clone_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1001: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_320, [0, 2, 1, 3]);  add_320 = None
    clone_330: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1001, memory_format = torch.contiguous_format);  permute_1001 = None
    view_1227: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_330, [1, 128, 1024]);  clone_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1228: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1227, [128, 1024]);  view_1227 = None
    mm_217: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1228, permute_1002);  permute_1002 = None
    permute_1003: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1228, [1, 0])
    mm_218: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1003, view_285);  permute_1003 = None
    permute_1004: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_218, [1, 0]);  mm_218 = None
    sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1228, [0], True);  view_1228 = None
    view_1229: "f32[1024]" = torch.ops.aten.reshape.default(sum_302, [1024]);  sum_302 = None
    permute_1005: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1004, [1, 0]);  permute_1004 = None
    view_1230: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_217, [1, 128, 1024]);  mm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1006: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_321, [0, 2, 1, 3]);  add_321 = None
    clone_331: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
    view_1231: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_331, [1, 128, 1024]);  clone_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1232: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1231, [128, 1024]);  view_1231 = None
    mm_219: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1232, permute_1007);  permute_1007 = None
    permute_1008: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1232, [1, 0])
    mm_220: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1008, view_285);  permute_1008 = None
    permute_1009: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_220, [1, 0]);  mm_220 = None
    sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1232, [0], True);  view_1232 = None
    view_1233: "f32[1024]" = torch.ops.aten.reshape.default(sum_303, [1024]);  sum_303 = None
    permute_1010: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1009, [1, 0]);  permute_1009 = None
    view_1234: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_219, [1, 128, 1024]);  mm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_322: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1230, view_1234);  view_1230 = view_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_462: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1226, 0.125);  view_1226 = None
    view_1235: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_462, [128, 1024]);  mul_462 = None
    mm_221: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1235, permute_1011);  permute_1011 = None
    permute_1012: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1235, [1, 0])
    mm_222: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1012, view_285);  permute_1012 = view_285 = None
    permute_1013: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_222, [1, 0]);  mm_222 = None
    sum_304: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1235, [0], True);  view_1235 = None
    view_1236: "f32[1024]" = torch.ops.aten.reshape.default(sum_304, [1024]);  sum_304 = None
    permute_1014: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1013, [1, 0]);  permute_1013 = None
    view_1237: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_221, [1, 128, 1024]);  mm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_323: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_322, view_1237);  add_322 = view_1237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_464: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_323, primals_223);  primals_223 = None
    mul_465: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_464, 1024)
    sum_305: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
    mul_466: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_464, mul_74);  mul_464 = None
    sum_306: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
    mul_467: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_74, sum_306);  sum_306 = None
    sub_223: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_465, sum_305);  mul_465 = sum_305 = None
    sub_224: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_223, mul_467);  sub_223 = mul_467 = None
    mul_468: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_71, sub_224);  div_71 = sub_224 = None
    mul_469: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_323, mul_74);  mul_74 = None
    sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
    sum_308: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_324: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_319, mul_468);  add_319 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:523, code: hidden_states = self.fc2(hidden_states)
    view_1238: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_324, [128, 1024])
    mm_223: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1238, permute_1015);  permute_1015 = None
    permute_1016: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1238, [1, 0])
    mm_224: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1016, view_283);  permute_1016 = view_283 = None
    permute_1017: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1238, [0], True);  view_1238 = None
    view_1239: "f32[1024]" = torch.ops.aten.reshape.default(sum_309, [1024]);  sum_309 = None
    permute_1018: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1017, [1, 0]);  permute_1017 = None
    view_1240: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_223, [1, 128, 4096]);  mm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:521, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_16: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_11, full_default_1, view_1240);  le_11 = view_1240 = None
    view_1241: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_16, [128, 4096]);  where_16 = None
    mm_225: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1241, permute_1019);  permute_1019 = None
    permute_1020: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1241, [1, 0])
    mm_226: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1020, view_281);  permute_1020 = view_281 = None
    permute_1021: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_226, [1, 0]);  mm_226 = None
    sum_310: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1241, [0], True);  view_1241 = None
    view_1242: "f32[4096]" = torch.ops.aten.reshape.default(sum_310, [4096]);  sum_310 = None
    permute_1022: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1021, [1, 0]);  permute_1021 = None
    view_1243: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_225, [1, 128, 1024]);  mm_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_471: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1243, primals_217);  primals_217 = None
    mul_472: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_471, 1024)
    sum_311: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
    mul_473: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_471, mul_72);  mul_471 = None
    sum_312: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    mul_474: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, sum_312);  sum_312 = None
    sub_226: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_472, sum_311);  mul_472 = sum_311 = None
    sub_227: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_226, mul_474);  sub_226 = mul_474 = None
    mul_475: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_227);  div_72 = sub_227 = None
    mul_476: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1243, mul_72);  mul_72 = None
    sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1243, [0, 1]);  view_1243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:520, code: hidden_states = self.final_layer_norm(hidden_states)
    add_325: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_324, mul_475);  add_324 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1244: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_325, [128, 1024])
    mm_227: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1244, permute_1023);  permute_1023 = None
    permute_1024: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1244, [1, 0])
    mm_228: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1024, view_279);  permute_1024 = view_279 = None
    permute_1025: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_228, [1, 0]);  mm_228 = None
    sum_315: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1244, [0], True);  view_1244 = None
    view_1245: "f32[1024]" = torch.ops.aten.reshape.default(sum_315, [1024]);  sum_315 = None
    permute_1026: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1025, [1, 0]);  permute_1025 = None
    view_1246: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_227, [1, 128, 1024]);  mm_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1247: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1246, [1, 128, 16, 64]);  view_1246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1027: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1247, [0, 2, 1, 3]);  view_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1248: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1027, [16, 128, 64]);  permute_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_160: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1028, view_1248);  permute_1028 = None
    bmm_161: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1248, permute_1029);  view_1248 = permute_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_477: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_161, div_13);  bmm_161 = None
    sum_316: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [-1], True)
    mul_478: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_13, sum_316);  div_13 = sum_316 = None
    sub_228: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_162: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1030, sub_228);  permute_1030 = None
    bmm_163: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_228, permute_1031);  sub_228 = permute_1031 = None
    permute_1032: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_162, [0, 2, 1]);  bmm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1249: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_160, [1, 16, 128, 64]);  bmm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_326: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_6, view_1249);  tangents_6 = view_1249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1250: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1032, [1, 16, 128, 64]);  permute_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_327: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_1250);  tangents_5 = view_1250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1251: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_163, [1, 16, 128, 64]);  bmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1033: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1251, [0, 2, 1, 3]);  view_1251 = None
    clone_332: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1033, memory_format = torch.contiguous_format);  permute_1033 = None
    view_1252: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_332, [1, 128, 1024]);  clone_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1034: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_326, [0, 2, 1, 3]);  add_326 = None
    clone_333: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1034, memory_format = torch.contiguous_format);  permute_1034 = None
    view_1253: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_333, [1, 128, 1024]);  clone_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_1254: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1253, [128, 1024]);  view_1253 = None
    mm_229: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1254, permute_1035);  permute_1035 = None
    permute_1036: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1254, [1, 0])
    mm_230: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1036, view_267);  permute_1036 = None
    permute_1037: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_230, [1, 0]);  mm_230 = None
    sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1254, [0], True);  view_1254 = None
    view_1255: "f32[1024]" = torch.ops.aten.reshape.default(sum_317, [1024]);  sum_317 = None
    permute_1038: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1037, [1, 0]);  permute_1037 = None
    view_1256: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_229, [1, 128, 1024]);  mm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:264, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    add_328: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_318, view_1256);  add_318 = view_1256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1039: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_327, [0, 2, 1, 3]);  add_327 = None
    clone_334: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1039, memory_format = torch.contiguous_format);  permute_1039 = None
    view_1257: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_334, [1, 128, 1024]);  clone_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_1258: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1257, [128, 1024]);  view_1257 = None
    mm_231: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1258, permute_1040);  permute_1040 = None
    permute_1041: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1258, [1, 0])
    mm_232: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1041, view_267);  permute_1041 = view_267 = None
    permute_1042: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1258, [0], True);  view_1258 = None
    view_1259: "f32[1024]" = torch.ops.aten.reshape.default(sum_318, [1024]);  sum_318 = None
    permute_1043: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1042, [1, 0]);  permute_1042 = None
    view_1260: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_231, [1, 128, 1024]);  mm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:263, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    add_329: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_328, view_1260);  add_328 = view_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_479: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1252, 0.125);  view_1252 = None
    view_1261: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_479, [128, 1024]);  mul_479 = None
    mm_233: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1261, permute_1044);  permute_1044 = None
    permute_1045: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1261, [1, 0])
    mm_234: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1045, view_265);  permute_1045 = view_265 = None
    permute_1046: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_234, [1, 0]);  mm_234 = None
    sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1261, [0], True);  view_1261 = None
    view_1262: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
    permute_1047: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1046, [1, 0]);  permute_1046 = None
    view_1263: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_233, [1, 128, 1024]);  mm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    mul_481: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1263, primals_207);  primals_207 = None
    mul_482: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_481, 1024)
    sum_320: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
    mul_483: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_481, mul_69);  mul_481 = None
    sum_321: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    mul_484: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_69, sum_321);  sum_321 = None
    sub_230: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_482, sum_320);  mul_482 = sum_320 = None
    sub_231: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_230, mul_484);  sub_230 = mul_484 = None
    mul_485: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_231);  div_73 = sub_231 = None
    mul_486: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1263, mul_69);  mul_69 = None
    sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
    sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1263, [0, 1]);  view_1263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:500, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    add_330: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_325, mul_485);  add_325 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1264: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_330, [128, 1024])
    mm_235: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1264, permute_1048);  permute_1048 = None
    permute_1049: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1264, [1, 0])
    mm_236: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1049, view_263);  permute_1049 = view_263 = None
    permute_1050: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_236, [1, 0]);  mm_236 = None
    sum_324: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1264, [0], True);  view_1264 = None
    view_1265: "f32[1024]" = torch.ops.aten.reshape.default(sum_324, [1024]);  sum_324 = None
    permute_1051: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1050, [1, 0]);  permute_1050 = None
    view_1266: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_235, [1, 128, 1024]);  mm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1267: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1266, [1, 128, 16, 64]);  view_1266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1052: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1267, [0, 2, 1, 3]);  view_1267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1268: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1052, [16, 128, 64]);  permute_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_164: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1053, view_1268);  permute_1053 = None
    bmm_165: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1268, permute_1054);  view_1268 = permute_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_487: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_165, alias_101);  bmm_165 = None
    sum_325: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [-1], True)
    mul_488: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_101, sum_325);  alias_101 = sum_325 = None
    sub_232: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:306, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_1269: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(sub_232, [1, 16, 128, 128]);  sub_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:305, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_1270: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(view_1269, [16, 128, 128]);  view_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_166: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1055, view_1270);  permute_1055 = None
    bmm_167: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(view_1270, permute_1056);  view_1270 = permute_1056 = None
    permute_1057: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_166, [0, 2, 1]);  bmm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1271: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_164, [1, 16, 128, 64]);  bmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    add_331: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_4, view_1271);  tangents_4 = view_1271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1272: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1057, [1, 16, 128, 64]);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    add_332: "f32[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_1272);  tangents_3 = view_1272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1273: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_167, [1, 16, 128, 64]);  bmm_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1058: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1273, [0, 2, 1, 3]);  view_1273 = None
    clone_335: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1058, memory_format = torch.contiguous_format);  permute_1058 = None
    view_1274: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_335, [1, 128, 1024]);  clone_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1059: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_331, [0, 2, 1, 3]);  add_331 = None
    clone_336: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
    view_1275: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_336, [1, 128, 1024]);  clone_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1276: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1275, [128, 1024]);  view_1275 = None
    mm_237: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1276, permute_1060);  permute_1060 = None
    permute_1061: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1276, [1, 0])
    mm_238: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1061, view_247);  permute_1061 = None
    permute_1062: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_238, [1, 0]);  mm_238 = None
    sum_326: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1276, [0], True);  view_1276 = None
    view_1277: "f32[1024]" = torch.ops.aten.reshape.default(sum_326, [1024]);  sum_326 = None
    permute_1063: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1062, [1, 0]);  permute_1062 = None
    view_1278: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_237, [1, 128, 1024]);  mm_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1064: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(add_332, [0, 2, 1, 3]);  add_332 = None
    clone_337: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1064, memory_format = torch.contiguous_format);  permute_1064 = None
    view_1279: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_337, [1, 128, 1024]);  clone_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1280: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1279, [128, 1024]);  view_1279 = None
    mm_239: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1280, permute_1065);  permute_1065 = None
    permute_1066: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1280, [1, 0])
    mm_240: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1066, view_247);  permute_1066 = None
    permute_1067: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_240, [1, 0]);  mm_240 = None
    sum_327: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1280, [0], True);  view_1280 = None
    view_1281: "f32[1024]" = torch.ops.aten.reshape.default(sum_327, [1024]);  sum_327 = None
    permute_1068: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1067, [1, 0]);  permute_1067 = None
    view_1282: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_239, [1, 128, 1024]);  mm_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_333: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1278, view_1282);  view_1278 = view_1282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_489: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1274, 0.125);  view_1274 = None
    view_1283: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_489, [128, 1024]);  mul_489 = None
    mm_241: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1283, permute_1069);  permute_1069 = None
    permute_1070: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1283, [1, 0])
    mm_242: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1070, view_247);  permute_1070 = view_247 = None
    permute_1071: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_242, [1, 0]);  mm_242 = None
    sum_328: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1283, [0], True);  view_1283 = None
    view_1284: "f32[1024]" = torch.ops.aten.reshape.default(sum_328, [1024]);  sum_328 = None
    permute_1072: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1071, [1, 0]);  permute_1071 = None
    view_1285: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_241, [1, 128, 1024]);  mm_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_334: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_333, view_1285);  add_333 = view_1285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_491: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_334, primals_197);  primals_197 = None
    mul_492: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_491, 1024)
    sum_329: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True)
    mul_493: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_491, mul_66);  mul_491 = None
    sum_330: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
    mul_494: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_66, sum_330);  sum_330 = None
    sub_234: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_492, sum_329);  mul_492 = sum_329 = None
    sub_235: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_234, mul_494);  sub_234 = mul_494 = None
    mul_495: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_235);  div_74 = sub_235 = None
    mul_496: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_334, mul_66);  mul_66 = None
    sum_331: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1]);  mul_496 = None
    sum_332: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_334, [0, 1]);  add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:479, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_335: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_330, mul_495);  add_330 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:1000, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    mul_497: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_335, 32.0);  add_335 = None
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view_243, 1)
    unsqueeze_6: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_17: "f32[1, 128, 1024]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_497);  unsqueeze_6 = mul_497 = None
    full_default_20: "f32[128112, 1024]" = torch.ops.aten.full.default([128112, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[128112, 1024]" = torch.ops.aten._unsafe_index_put.default(full_default_20, [view_243], where_17, True);  view_243 = where_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:852, code: hidden_states = self.layer_norm(hidden_states)
    mul_499: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_329, primals_194);  primals_194 = None
    mul_500: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_499, 1024)
    sum_333: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
    mul_501: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_499, mul_62);  mul_499 = None
    sum_334: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
    mul_502: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_62, sum_334);  sum_334 = None
    sub_237: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_500, sum_333);  mul_500 = sum_333 = None
    sub_238: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_237, mul_502);  sub_237 = mul_502 = None
    mul_503: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_238);  div_75 = sub_238 = None
    mul_504: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_329, mul_62);  mul_62 = None
    sum_335: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
    sum_336: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1286: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_503, [128, 1024])
    mm_243: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1286, permute_1073);  permute_1073 = None
    permute_1074: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1286, [1, 0])
    mm_244: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1074, view_241);  permute_1074 = view_241 = None
    permute_1075: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    sum_337: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1286, [0], True);  view_1286 = None
    view_1287: "f32[1024]" = torch.ops.aten.reshape.default(sum_337, [1024]);  sum_337 = None
    permute_1076: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1075, [1, 0]);  permute_1075 = None
    view_1288: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_18: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_12, full_default_1, view_1288);  le_12 = view_1288 = None
    view_1289: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_18, [128, 4096]);  where_18 = None
    mm_245: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1289, permute_1077);  permute_1077 = None
    permute_1078: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1289, [1, 0])
    mm_246: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1078, view_239);  permute_1078 = view_239 = None
    permute_1079: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_246, [1, 0]);  mm_246 = None
    sum_338: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
    view_1290: "f32[4096]" = torch.ops.aten.reshape.default(sum_338, [4096]);  sum_338 = None
    permute_1080: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1079, [1, 0]);  permute_1079 = None
    view_1291: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_245, [1, 128, 1024]);  mm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_506: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1291, primals_188);  primals_188 = None
    mul_507: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_506, 1024)
    sum_339: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
    mul_508: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_506, mul_60);  mul_506 = None
    sum_340: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
    mul_509: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, sum_340);  sum_340 = None
    sub_240: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_507, sum_339);  mul_507 = sum_339 = None
    sub_241: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_240, mul_509);  sub_240 = mul_509 = None
    mul_510: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_76, sub_241);  div_76 = sub_241 = None
    mul_511: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1291, mul_60);  mul_60 = None
    sum_341: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
    sum_342: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1291, [0, 1]);  view_1291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_336: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_503, mul_510);  mul_503 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1292: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_336, [128, 1024])
    mm_247: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1292, permute_1081);  permute_1081 = None
    permute_1082: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1292, [1, 0])
    mm_248: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1082, view_237);  permute_1082 = view_237 = None
    permute_1083: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_248, [1, 0]);  mm_248 = None
    sum_343: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1292, [0], True);  view_1292 = None
    view_1293: "f32[1024]" = torch.ops.aten.reshape.default(sum_343, [1024]);  sum_343 = None
    permute_1084: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1083, [1, 0]);  permute_1083 = None
    view_1294: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_247, [1, 128, 1024]);  mm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1295: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1294, [1, 128, 16, 64]);  view_1294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1085: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1295, [0, 2, 1, 3]);  view_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1296: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1085, [16, 128, 64]);  permute_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_168: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1086, view_1296);  permute_1086 = None
    bmm_169: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1296, permute_1087);  view_1296 = permute_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_512: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_169, div_11);  bmm_169 = None
    sum_344: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [-1], True)
    mul_513: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_11, sum_344);  div_11 = sum_344 = None
    sub_242: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_170: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1088, sub_242);  permute_1088 = None
    bmm_171: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_242, permute_1089);  sub_242 = permute_1089 = None
    permute_1090: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_170, [0, 2, 1]);  bmm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1297: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_168, [1, 16, 128, 64]);  bmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1298: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1090, [1, 16, 128, 64]);  permute_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1299: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_171, [1, 16, 128, 64]);  bmm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1091: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1299, [0, 2, 1, 3]);  view_1299 = None
    clone_338: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1091, memory_format = torch.contiguous_format);  permute_1091 = None
    view_1300: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_338, [1, 128, 1024]);  clone_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1092: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1297, [0, 2, 1, 3]);  view_1297 = None
    clone_339: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
    view_1301: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_339, [1, 128, 1024]);  clone_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1302: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1301, [128, 1024]);  view_1301 = None
    mm_249: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1302, permute_1093);  permute_1093 = None
    permute_1094: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1302, [1, 0])
    mm_250: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1094, view_223);  permute_1094 = None
    permute_1095: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_250, [1, 0]);  mm_250 = None
    sum_345: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1302, [0], True);  view_1302 = None
    view_1303: "f32[1024]" = torch.ops.aten.reshape.default(sum_345, [1024]);  sum_345 = None
    permute_1096: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1095, [1, 0]);  permute_1095 = None
    view_1304: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_249, [1, 128, 1024]);  mm_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1097: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1298, [0, 2, 1, 3]);  view_1298 = None
    view_1305: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1097, [1, 128, 1024]);  permute_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1306: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1305, [128, 1024]);  view_1305 = None
    mm_251: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1306, permute_1098);  permute_1098 = None
    permute_1099: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1306, [1, 0])
    mm_252: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1099, view_223);  permute_1099 = None
    permute_1100: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_252, [1, 0]);  mm_252 = None
    sum_346: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1306, [0], True);  view_1306 = None
    view_1307: "f32[1024]" = torch.ops.aten.reshape.default(sum_346, [1024]);  sum_346 = None
    permute_1101: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1100, [1, 0]);  permute_1100 = None
    view_1308: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_251, [1, 128, 1024]);  mm_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_337: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1304, view_1308);  view_1304 = view_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_514: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1300, 0.125);  view_1300 = None
    view_1309: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_514, [128, 1024]);  mul_514 = None
    mm_253: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1309, permute_1102);  permute_1102 = None
    permute_1103: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1309, [1, 0])
    mm_254: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1103, view_223);  permute_1103 = view_223 = None
    permute_1104: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_254, [1, 0]);  mm_254 = None
    sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1309, [0], True);  view_1309 = None
    view_1310: "f32[1024]" = torch.ops.aten.reshape.default(sum_347, [1024]);  sum_347 = None
    permute_1105: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1104, [1, 0]);  permute_1104 = None
    view_1311: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_253, [1, 128, 1024]);  mm_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_338: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_337, view_1311);  add_337 = view_1311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_516: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_338, primals_178);  primals_178 = None
    mul_517: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_516, 1024)
    sum_348: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
    mul_518: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_516, mul_57);  mul_516 = None
    sum_349: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
    mul_519: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, sum_349);  sum_349 = None
    sub_244: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_517, sum_348);  mul_517 = sum_348 = None
    sub_245: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_244, mul_519);  sub_244 = mul_519 = None
    mul_520: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_77, sub_245);  div_77 = sub_245 = None
    mul_521: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_57);  mul_57 = None
    sum_350: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1]);  mul_521 = None
    sum_351: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_339: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_336, mul_520);  add_336 = mul_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1312: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_339, [128, 1024])
    mm_255: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1312, permute_1106);  permute_1106 = None
    permute_1107: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1312, [1, 0])
    mm_256: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1107, view_221);  permute_1107 = view_221 = None
    permute_1108: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    sum_352: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1312, [0], True);  view_1312 = None
    view_1313: "f32[1024]" = torch.ops.aten.reshape.default(sum_352, [1024]);  sum_352 = None
    permute_1109: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1108, [1, 0]);  permute_1108 = None
    view_1314: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_19: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_13, full_default_1, view_1314);  le_13 = view_1314 = None
    view_1315: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_19, [128, 4096]);  where_19 = None
    mm_257: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1315, permute_1110);  permute_1110 = None
    permute_1111: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1315, [1, 0])
    mm_258: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1111, view_219);  permute_1111 = view_219 = None
    permute_1112: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_258, [1, 0]);  mm_258 = None
    sum_353: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1315, [0], True);  view_1315 = None
    view_1316: "f32[4096]" = torch.ops.aten.reshape.default(sum_353, [4096]);  sum_353 = None
    permute_1113: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1112, [1, 0]);  permute_1112 = None
    view_1317: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_257, [1, 128, 1024]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_523: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1317, primals_172);  primals_172 = None
    mul_524: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_523, 1024)
    sum_354: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
    mul_525: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_523, mul_55);  mul_523 = None
    sum_355: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
    mul_526: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_55, sum_355);  sum_355 = None
    sub_247: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_524, sum_354);  mul_524 = sum_354 = None
    sub_248: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_247, mul_526);  sub_247 = mul_526 = None
    mul_527: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_78, sub_248);  div_78 = sub_248 = None
    mul_528: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1317, mul_55);  mul_55 = None
    sum_356: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
    sum_357: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1317, [0, 1]);  view_1317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_340: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_339, mul_527);  add_339 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1318: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_340, [128, 1024])
    mm_259: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1318, permute_1114);  permute_1114 = None
    permute_1115: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1318, [1, 0])
    mm_260: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1115, view_217);  permute_1115 = view_217 = None
    permute_1116: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_260, [1, 0]);  mm_260 = None
    sum_358: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1318, [0], True);  view_1318 = None
    view_1319: "f32[1024]" = torch.ops.aten.reshape.default(sum_358, [1024]);  sum_358 = None
    permute_1117: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1116, [1, 0]);  permute_1116 = None
    view_1320: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_259, [1, 128, 1024]);  mm_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1321: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1320, [1, 128, 16, 64]);  view_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1118: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1321, [0, 2, 1, 3]);  view_1321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1322: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1118, [16, 128, 64]);  permute_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_172: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1119, view_1322);  permute_1119 = None
    bmm_173: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1322, permute_1120);  view_1322 = permute_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_529: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_173, div_10);  bmm_173 = None
    sum_359: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [-1], True)
    mul_530: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_10, sum_359);  div_10 = sum_359 = None
    sub_249: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_174: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1121, sub_249);  permute_1121 = None
    bmm_175: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_249, permute_1122);  sub_249 = permute_1122 = None
    permute_1123: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_174, [0, 2, 1]);  bmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1323: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_172, [1, 16, 128, 64]);  bmm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1324: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1123, [1, 16, 128, 64]);  permute_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1325: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_175, [1, 16, 128, 64]);  bmm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1124: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1325, [0, 2, 1, 3]);  view_1325 = None
    clone_340: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1124, memory_format = torch.contiguous_format);  permute_1124 = None
    view_1326: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_340, [1, 128, 1024]);  clone_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1125: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1323, [0, 2, 1, 3]);  view_1323 = None
    clone_341: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
    view_1327: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_341, [1, 128, 1024]);  clone_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1328: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1327, [128, 1024]);  view_1327 = None
    mm_261: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1328, permute_1126);  permute_1126 = None
    permute_1127: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1328, [1, 0])
    mm_262: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1127, view_203);  permute_1127 = None
    permute_1128: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_262, [1, 0]);  mm_262 = None
    sum_360: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1328, [0], True);  view_1328 = None
    view_1329: "f32[1024]" = torch.ops.aten.reshape.default(sum_360, [1024]);  sum_360 = None
    permute_1129: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1128, [1, 0]);  permute_1128 = None
    view_1330: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_261, [1, 128, 1024]);  mm_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1130: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1324, [0, 2, 1, 3]);  view_1324 = None
    view_1331: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1130, [1, 128, 1024]);  permute_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1332: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1331, [128, 1024]);  view_1331 = None
    mm_263: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1332, permute_1131);  permute_1131 = None
    permute_1132: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1332, [1, 0])
    mm_264: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1132, view_203);  permute_1132 = None
    permute_1133: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_264, [1, 0]);  mm_264 = None
    sum_361: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1332, [0], True);  view_1332 = None
    view_1333: "f32[1024]" = torch.ops.aten.reshape.default(sum_361, [1024]);  sum_361 = None
    permute_1134: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1133, [1, 0]);  permute_1133 = None
    view_1334: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_263, [1, 128, 1024]);  mm_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_341: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1330, view_1334);  view_1330 = view_1334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_531: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1326, 0.125);  view_1326 = None
    view_1335: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_531, [128, 1024]);  mul_531 = None
    mm_265: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1335, permute_1135);  permute_1135 = None
    permute_1136: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1335, [1, 0])
    mm_266: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1136, view_203);  permute_1136 = view_203 = None
    permute_1137: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_266, [1, 0]);  mm_266 = None
    sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1335, [0], True);  view_1335 = None
    view_1336: "f32[1024]" = torch.ops.aten.reshape.default(sum_362, [1024]);  sum_362 = None
    permute_1138: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1137, [1, 0]);  permute_1137 = None
    view_1337: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_265, [1, 128, 1024]);  mm_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_342: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_341, view_1337);  add_341 = view_1337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_533: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_342, primals_162);  primals_162 = None
    mul_534: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_533, 1024)
    sum_363: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True)
    mul_535: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_533, mul_52);  mul_533 = None
    sum_364: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True);  mul_535 = None
    mul_536: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, sum_364);  sum_364 = None
    sub_251: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_534, sum_363);  mul_534 = sum_363 = None
    sub_252: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_251, mul_536);  sub_251 = mul_536 = None
    mul_537: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_79, sub_252);  div_79 = sub_252 = None
    mul_538: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_342, mul_52);  mul_52 = None
    sum_365: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 1]);  mul_538 = None
    sum_366: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_342, [0, 1]);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_343: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_340, mul_537);  add_340 = mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1338: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_343, [128, 1024])
    mm_267: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1338, permute_1139);  permute_1139 = None
    permute_1140: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1338, [1, 0])
    mm_268: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1140, view_201);  permute_1140 = view_201 = None
    permute_1141: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    sum_367: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1338, [0], True);  view_1338 = None
    view_1339: "f32[1024]" = torch.ops.aten.reshape.default(sum_367, [1024]);  sum_367 = None
    permute_1142: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1141, [1, 0]);  permute_1141 = None
    view_1340: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_20: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_14, full_default_1, view_1340);  le_14 = view_1340 = None
    view_1341: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_20, [128, 4096]);  where_20 = None
    mm_269: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1341, permute_1143);  permute_1143 = None
    permute_1144: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1341, [1, 0])
    mm_270: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1144, view_199);  permute_1144 = view_199 = None
    permute_1145: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_270, [1, 0]);  mm_270 = None
    sum_368: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1341, [0], True);  view_1341 = None
    view_1342: "f32[4096]" = torch.ops.aten.reshape.default(sum_368, [4096]);  sum_368 = None
    permute_1146: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1145, [1, 0]);  permute_1145 = None
    view_1343: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_269, [1, 128, 1024]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_540: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1343, primals_156);  primals_156 = None
    mul_541: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_540, 1024)
    sum_369: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [2], True)
    mul_542: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_540, mul_50);  mul_540 = None
    sum_370: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2], True);  mul_542 = None
    mul_543: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_50, sum_370);  sum_370 = None
    sub_254: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_541, sum_369);  mul_541 = sum_369 = None
    sub_255: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_254, mul_543);  sub_254 = mul_543 = None
    mul_544: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_80, sub_255);  div_80 = sub_255 = None
    mul_545: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1343, mul_50);  mul_50 = None
    sum_371: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 1]);  mul_545 = None
    sum_372: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1343, [0, 1]);  view_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_344: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_343, mul_544);  add_343 = mul_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1344: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_344, [128, 1024])
    mm_271: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1344, permute_1147);  permute_1147 = None
    permute_1148: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1344, [1, 0])
    mm_272: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1148, view_197);  permute_1148 = view_197 = None
    permute_1149: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_272, [1, 0]);  mm_272 = None
    sum_373: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1344, [0], True);  view_1344 = None
    view_1345: "f32[1024]" = torch.ops.aten.reshape.default(sum_373, [1024]);  sum_373 = None
    permute_1150: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1149, [1, 0]);  permute_1149 = None
    view_1346: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_271, [1, 128, 1024]);  mm_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1347: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1346, [1, 128, 16, 64]);  view_1346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1151: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1347, [0, 2, 1, 3]);  view_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1348: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1151, [16, 128, 64]);  permute_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_176: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1152, view_1348);  permute_1152 = None
    bmm_177: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1348, permute_1153);  view_1348 = permute_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_546: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_177, div_9);  bmm_177 = None
    sum_374: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [-1], True)
    mul_547: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_9, sum_374);  div_9 = sum_374 = None
    sub_256: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_178: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1154, sub_256);  permute_1154 = None
    bmm_179: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_256, permute_1155);  sub_256 = permute_1155 = None
    permute_1156: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_178, [0, 2, 1]);  bmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1349: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_176, [1, 16, 128, 64]);  bmm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1350: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1156, [1, 16, 128, 64]);  permute_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1351: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_179, [1, 16, 128, 64]);  bmm_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1157: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1351, [0, 2, 1, 3]);  view_1351 = None
    clone_342: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1157, memory_format = torch.contiguous_format);  permute_1157 = None
    view_1352: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_342, [1, 128, 1024]);  clone_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1158: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1349, [0, 2, 1, 3]);  view_1349 = None
    clone_343: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
    view_1353: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_343, [1, 128, 1024]);  clone_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1354: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1353, [128, 1024]);  view_1353 = None
    mm_273: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1354, permute_1159);  permute_1159 = None
    permute_1160: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1354, [1, 0])
    mm_274: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1160, view_183);  permute_1160 = None
    permute_1161: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_274, [1, 0]);  mm_274 = None
    sum_375: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1354, [0], True);  view_1354 = None
    view_1355: "f32[1024]" = torch.ops.aten.reshape.default(sum_375, [1024]);  sum_375 = None
    permute_1162: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1161, [1, 0]);  permute_1161 = None
    view_1356: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_273, [1, 128, 1024]);  mm_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1163: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1350, [0, 2, 1, 3]);  view_1350 = None
    view_1357: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1163, [1, 128, 1024]);  permute_1163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1358: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1357, [128, 1024]);  view_1357 = None
    mm_275: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1358, permute_1164);  permute_1164 = None
    permute_1165: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1358, [1, 0])
    mm_276: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1165, view_183);  permute_1165 = None
    permute_1166: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_276, [1, 0]);  mm_276 = None
    sum_376: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1358, [0], True);  view_1358 = None
    view_1359: "f32[1024]" = torch.ops.aten.reshape.default(sum_376, [1024]);  sum_376 = None
    permute_1167: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1166, [1, 0]);  permute_1166 = None
    view_1360: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_275, [1, 128, 1024]);  mm_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_345: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1356, view_1360);  view_1356 = view_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_548: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1352, 0.125);  view_1352 = None
    view_1361: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_548, [128, 1024]);  mul_548 = None
    mm_277: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1361, permute_1168);  permute_1168 = None
    permute_1169: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1361, [1, 0])
    mm_278: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1169, view_183);  permute_1169 = view_183 = None
    permute_1170: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_278, [1, 0]);  mm_278 = None
    sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1361, [0], True);  view_1361 = None
    view_1362: "f32[1024]" = torch.ops.aten.reshape.default(sum_377, [1024]);  sum_377 = None
    permute_1171: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1170, [1, 0]);  permute_1170 = None
    view_1363: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_277, [1, 128, 1024]);  mm_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_346: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_345, view_1363);  add_345 = view_1363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_550: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_346, primals_146);  primals_146 = None
    mul_551: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_550, 1024)
    sum_378: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True)
    mul_552: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_550, mul_47);  mul_550 = None
    sum_379: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True);  mul_552 = None
    mul_553: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_47, sum_379);  sum_379 = None
    sub_258: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_551, sum_378);  mul_551 = sum_378 = None
    sub_259: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_258, mul_553);  sub_258 = mul_553 = None
    mul_554: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_81, sub_259);  div_81 = sub_259 = None
    mul_555: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_346, mul_47);  mul_47 = None
    sum_380: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 1]);  mul_555 = None
    sum_381: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_347: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_344, mul_554);  add_344 = mul_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1364: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_347, [128, 1024])
    mm_279: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1364, permute_1172);  permute_1172 = None
    permute_1173: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1364, [1, 0])
    mm_280: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1173, view_181);  permute_1173 = view_181 = None
    permute_1174: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_280, [1, 0]);  mm_280 = None
    sum_382: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1364, [0], True);  view_1364 = None
    view_1365: "f32[1024]" = torch.ops.aten.reshape.default(sum_382, [1024]);  sum_382 = None
    permute_1175: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1174, [1, 0]);  permute_1174 = None
    view_1366: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_21: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_15, full_default_1, view_1366);  le_15 = view_1366 = None
    view_1367: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_21, [128, 4096]);  where_21 = None
    mm_281: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1367, permute_1176);  permute_1176 = None
    permute_1177: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1367, [1, 0])
    mm_282: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1177, view_179);  permute_1177 = view_179 = None
    permute_1178: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_282, [1, 0]);  mm_282 = None
    sum_383: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1367, [0], True);  view_1367 = None
    view_1368: "f32[4096]" = torch.ops.aten.reshape.default(sum_383, [4096]);  sum_383 = None
    permute_1179: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1178, [1, 0]);  permute_1178 = None
    view_1369: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_281, [1, 128, 1024]);  mm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_557: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1369, primals_140);  primals_140 = None
    mul_558: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_557, 1024)
    sum_384: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
    mul_559: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_557, mul_45);  mul_557 = None
    sum_385: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
    mul_560: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_45, sum_385);  sum_385 = None
    sub_261: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_558, sum_384);  mul_558 = sum_384 = None
    sub_262: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_261, mul_560);  sub_261 = mul_560 = None
    mul_561: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_82, sub_262);  div_82 = sub_262 = None
    mul_562: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1369, mul_45);  mul_45 = None
    sum_386: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
    sum_387: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1369, [0, 1]);  view_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_348: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_347, mul_561);  add_347 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1370: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_348, [128, 1024])
    mm_283: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1370, permute_1180);  permute_1180 = None
    permute_1181: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1370, [1, 0])
    mm_284: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1181, view_177);  permute_1181 = view_177 = None
    permute_1182: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_284, [1, 0]);  mm_284 = None
    sum_388: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1370, [0], True);  view_1370 = None
    view_1371: "f32[1024]" = torch.ops.aten.reshape.default(sum_388, [1024]);  sum_388 = None
    permute_1183: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1182, [1, 0]);  permute_1182 = None
    view_1372: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_283, [1, 128, 1024]);  mm_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1373: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1372, [1, 128, 16, 64]);  view_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1184: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1373, [0, 2, 1, 3]);  view_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1374: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1184, [16, 128, 64]);  permute_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_180: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1185, view_1374);  permute_1185 = None
    bmm_181: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1374, permute_1186);  view_1374 = permute_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_563: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_181, div_8);  bmm_181 = None
    sum_389: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [-1], True)
    mul_564: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_8, sum_389);  div_8 = sum_389 = None
    sub_263: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_182: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1187, sub_263);  permute_1187 = None
    bmm_183: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_263, permute_1188);  sub_263 = permute_1188 = None
    permute_1189: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_182, [0, 2, 1]);  bmm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1375: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_180, [1, 16, 128, 64]);  bmm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1376: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1189, [1, 16, 128, 64]);  permute_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1377: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_183, [1, 16, 128, 64]);  bmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1190: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1377, [0, 2, 1, 3]);  view_1377 = None
    clone_344: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1190, memory_format = torch.contiguous_format);  permute_1190 = None
    view_1378: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_344, [1, 128, 1024]);  clone_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1191: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1375, [0, 2, 1, 3]);  view_1375 = None
    clone_345: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
    view_1379: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_345, [1, 128, 1024]);  clone_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1380: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1379, [128, 1024]);  view_1379 = None
    mm_285: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1380, permute_1192);  permute_1192 = None
    permute_1193: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1380, [1, 0])
    mm_286: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1193, view_163);  permute_1193 = None
    permute_1194: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_286, [1, 0]);  mm_286 = None
    sum_390: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1380, [0], True);  view_1380 = None
    view_1381: "f32[1024]" = torch.ops.aten.reshape.default(sum_390, [1024]);  sum_390 = None
    permute_1195: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1194, [1, 0]);  permute_1194 = None
    view_1382: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_285, [1, 128, 1024]);  mm_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1196: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1376, [0, 2, 1, 3]);  view_1376 = None
    view_1383: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1196, [1, 128, 1024]);  permute_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1384: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1383, [128, 1024]);  view_1383 = None
    mm_287: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1384, permute_1197);  permute_1197 = None
    permute_1198: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1384, [1, 0])
    mm_288: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1198, view_163);  permute_1198 = None
    permute_1199: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_288, [1, 0]);  mm_288 = None
    sum_391: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1384, [0], True);  view_1384 = None
    view_1385: "f32[1024]" = torch.ops.aten.reshape.default(sum_391, [1024]);  sum_391 = None
    permute_1200: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1199, [1, 0]);  permute_1199 = None
    view_1386: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_287, [1, 128, 1024]);  mm_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_349: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1382, view_1386);  view_1382 = view_1386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_565: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1378, 0.125);  view_1378 = None
    view_1387: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_565, [128, 1024]);  mul_565 = None
    mm_289: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1387, permute_1201);  permute_1201 = None
    permute_1202: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1387, [1, 0])
    mm_290: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1202, view_163);  permute_1202 = view_163 = None
    permute_1203: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_290, [1, 0]);  mm_290 = None
    sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1387, [0], True);  view_1387 = None
    view_1388: "f32[1024]" = torch.ops.aten.reshape.default(sum_392, [1024]);  sum_392 = None
    permute_1204: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1203, [1, 0]);  permute_1203 = None
    view_1389: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_289, [1, 128, 1024]);  mm_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_350: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_349, view_1389);  add_349 = view_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_567: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_350, primals_130);  primals_130 = None
    mul_568: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_567, 1024)
    sum_393: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True)
    mul_569: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_567, mul_42);  mul_567 = None
    sum_394: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True);  mul_569 = None
    mul_570: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_394);  sum_394 = None
    sub_265: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_568, sum_393);  mul_568 = sum_393 = None
    sub_266: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_265, mul_570);  sub_265 = mul_570 = None
    mul_571: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_83, sub_266);  div_83 = sub_266 = None
    mul_572: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_350, mul_42);  mul_42 = None
    sum_395: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 1]);  mul_572 = None
    sum_396: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 1]);  add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_351: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_348, mul_571);  add_348 = mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1390: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_351, [128, 1024])
    mm_291: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1390, permute_1205);  permute_1205 = None
    permute_1206: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1390, [1, 0])
    mm_292: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1206, view_161);  permute_1206 = view_161 = None
    permute_1207: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_292, [1, 0]);  mm_292 = None
    sum_397: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1390, [0], True);  view_1390 = None
    view_1391: "f32[1024]" = torch.ops.aten.reshape.default(sum_397, [1024]);  sum_397 = None
    permute_1208: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1207, [1, 0]);  permute_1207 = None
    view_1392: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_22: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_16, full_default_1, view_1392);  le_16 = view_1392 = None
    view_1393: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_22, [128, 4096]);  where_22 = None
    mm_293: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1393, permute_1209);  permute_1209 = None
    permute_1210: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1393, [1, 0])
    mm_294: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1210, view_159);  permute_1210 = view_159 = None
    permute_1211: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_294, [1, 0]);  mm_294 = None
    sum_398: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1393, [0], True);  view_1393 = None
    view_1394: "f32[4096]" = torch.ops.aten.reshape.default(sum_398, [4096]);  sum_398 = None
    permute_1212: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1211, [1, 0]);  permute_1211 = None
    view_1395: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_293, [1, 128, 1024]);  mm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_574: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1395, primals_124);  primals_124 = None
    mul_575: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_574, 1024)
    sum_399: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True)
    mul_576: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_574, mul_40);  mul_574 = None
    sum_400: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True);  mul_576 = None
    mul_577: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_40, sum_400);  sum_400 = None
    sub_268: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_575, sum_399);  mul_575 = sum_399 = None
    sub_269: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_268, mul_577);  sub_268 = mul_577 = None
    mul_578: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_84, sub_269);  div_84 = sub_269 = None
    mul_579: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1395, mul_40);  mul_40 = None
    sum_401: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None
    sum_402: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1395, [0, 1]);  view_1395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_352: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_351, mul_578);  add_351 = mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1396: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_352, [128, 1024])
    mm_295: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1396, permute_1213);  permute_1213 = None
    permute_1214: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1396, [1, 0])
    mm_296: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1214, view_157);  permute_1214 = view_157 = None
    permute_1215: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_296, [1, 0]);  mm_296 = None
    sum_403: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1396, [0], True);  view_1396 = None
    view_1397: "f32[1024]" = torch.ops.aten.reshape.default(sum_403, [1024]);  sum_403 = None
    permute_1216: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1215, [1, 0]);  permute_1215 = None
    view_1398: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_295, [1, 128, 1024]);  mm_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1399: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1398, [1, 128, 16, 64]);  view_1398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1217: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1399, [0, 2, 1, 3]);  view_1399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1400: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1217, [16, 128, 64]);  permute_1217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_184: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1218, view_1400);  permute_1218 = None
    bmm_185: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1400, permute_1219);  view_1400 = permute_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_580: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_185, div_7);  bmm_185 = None
    sum_404: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_580, [-1], True)
    mul_581: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_7, sum_404);  div_7 = sum_404 = None
    sub_270: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_186: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1220, sub_270);  permute_1220 = None
    bmm_187: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_270, permute_1221);  sub_270 = permute_1221 = None
    permute_1222: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_186, [0, 2, 1]);  bmm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1401: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_184, [1, 16, 128, 64]);  bmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1402: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1222, [1, 16, 128, 64]);  permute_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1403: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_187, [1, 16, 128, 64]);  bmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1223: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1403, [0, 2, 1, 3]);  view_1403 = None
    clone_346: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1223, memory_format = torch.contiguous_format);  permute_1223 = None
    view_1404: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_346, [1, 128, 1024]);  clone_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1224: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1401, [0, 2, 1, 3]);  view_1401 = None
    clone_347: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
    view_1405: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_347, [1, 128, 1024]);  clone_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1406: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1405, [128, 1024]);  view_1405 = None
    mm_297: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1406, permute_1225);  permute_1225 = None
    permute_1226: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1406, [1, 0])
    mm_298: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1226, view_143);  permute_1226 = None
    permute_1227: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_298, [1, 0]);  mm_298 = None
    sum_405: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1406, [0], True);  view_1406 = None
    view_1407: "f32[1024]" = torch.ops.aten.reshape.default(sum_405, [1024]);  sum_405 = None
    permute_1228: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1227, [1, 0]);  permute_1227 = None
    view_1408: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_297, [1, 128, 1024]);  mm_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1229: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None
    view_1409: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1229, [1, 128, 1024]);  permute_1229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1410: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1409, [128, 1024]);  view_1409 = None
    mm_299: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1410, permute_1230);  permute_1230 = None
    permute_1231: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1410, [1, 0])
    mm_300: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1231, view_143);  permute_1231 = None
    permute_1232: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_300, [1, 0]);  mm_300 = None
    sum_406: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1410, [0], True);  view_1410 = None
    view_1411: "f32[1024]" = torch.ops.aten.reshape.default(sum_406, [1024]);  sum_406 = None
    permute_1233: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1232, [1, 0]);  permute_1232 = None
    view_1412: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_299, [1, 128, 1024]);  mm_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_353: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1408, view_1412);  view_1408 = view_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_582: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1404, 0.125);  view_1404 = None
    view_1413: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_582, [128, 1024]);  mul_582 = None
    mm_301: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1413, permute_1234);  permute_1234 = None
    permute_1235: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1413, [1, 0])
    mm_302: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1235, view_143);  permute_1235 = view_143 = None
    permute_1236: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_302, [1, 0]);  mm_302 = None
    sum_407: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1413, [0], True);  view_1413 = None
    view_1414: "f32[1024]" = torch.ops.aten.reshape.default(sum_407, [1024]);  sum_407 = None
    permute_1237: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1236, [1, 0]);  permute_1236 = None
    view_1415: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_301, [1, 128, 1024]);  mm_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_354: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_353, view_1415);  add_353 = view_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_584: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_354, primals_114);  primals_114 = None
    mul_585: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_584, 1024)
    sum_408: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True)
    mul_586: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_584, mul_37);  mul_584 = None
    sum_409: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True);  mul_586 = None
    mul_587: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_37, sum_409);  sum_409 = None
    sub_272: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_585, sum_408);  mul_585 = sum_408 = None
    sub_273: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_272, mul_587);  sub_272 = mul_587 = None
    mul_588: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_85, sub_273);  div_85 = sub_273 = None
    mul_589: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_354, mul_37);  mul_37 = None
    sum_410: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 1]);  mul_589 = None
    sum_411: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_355: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_352, mul_588);  add_352 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1416: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_355, [128, 1024])
    mm_303: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1416, permute_1238);  permute_1238 = None
    permute_1239: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1416, [1, 0])
    mm_304: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1239, view_141);  permute_1239 = view_141 = None
    permute_1240: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_304, [1, 0]);  mm_304 = None
    sum_412: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1416, [0], True);  view_1416 = None
    view_1417: "f32[1024]" = torch.ops.aten.reshape.default(sum_412, [1024]);  sum_412 = None
    permute_1241: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1240, [1, 0]);  permute_1240 = None
    view_1418: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_23: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_17, full_default_1, view_1418);  le_17 = view_1418 = None
    view_1419: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_23, [128, 4096]);  where_23 = None
    mm_305: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1419, permute_1242);  permute_1242 = None
    permute_1243: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1419, [1, 0])
    mm_306: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1243, view_139);  permute_1243 = view_139 = None
    permute_1244: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_306, [1, 0]);  mm_306 = None
    sum_413: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1419, [0], True);  view_1419 = None
    view_1420: "f32[4096]" = torch.ops.aten.reshape.default(sum_413, [4096]);  sum_413 = None
    permute_1245: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1244, [1, 0]);  permute_1244 = None
    view_1421: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_305, [1, 128, 1024]);  mm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_591: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1421, primals_108);  primals_108 = None
    mul_592: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_591, 1024)
    sum_414: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [2], True)
    mul_593: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_591, mul_35);  mul_591 = None
    sum_415: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
    mul_594: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_35, sum_415);  sum_415 = None
    sub_275: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_592, sum_414);  mul_592 = sum_414 = None
    sub_276: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_275, mul_594);  sub_275 = mul_594 = None
    mul_595: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_86, sub_276);  div_86 = sub_276 = None
    mul_596: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1421, mul_35);  mul_35 = None
    sum_416: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1]);  mul_596 = None
    sum_417: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1421, [0, 1]);  view_1421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_356: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_355, mul_595);  add_355 = mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1422: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_356, [128, 1024])
    mm_307: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1422, permute_1246);  permute_1246 = None
    permute_1247: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1422, [1, 0])
    mm_308: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1247, view_137);  permute_1247 = view_137 = None
    permute_1248: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_308, [1, 0]);  mm_308 = None
    sum_418: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1422, [0], True);  view_1422 = None
    view_1423: "f32[1024]" = torch.ops.aten.reshape.default(sum_418, [1024]);  sum_418 = None
    permute_1249: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1248, [1, 0]);  permute_1248 = None
    view_1424: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_307, [1, 128, 1024]);  mm_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1425: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1424, [1, 128, 16, 64]);  view_1424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1250: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1425, [0, 2, 1, 3]);  view_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1426: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1250, [16, 128, 64]);  permute_1250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_188: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1251, view_1426);  permute_1251 = None
    bmm_189: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1426, permute_1252);  view_1426 = permute_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_597: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_189, div_6);  bmm_189 = None
    sum_419: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [-1], True)
    mul_598: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_6, sum_419);  div_6 = sum_419 = None
    sub_277: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_190: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1253, sub_277);  permute_1253 = None
    bmm_191: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_277, permute_1254);  sub_277 = permute_1254 = None
    permute_1255: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_190, [0, 2, 1]);  bmm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1427: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_188, [1, 16, 128, 64]);  bmm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1428: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1255, [1, 16, 128, 64]);  permute_1255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1429: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_191, [1, 16, 128, 64]);  bmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1256: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1429, [0, 2, 1, 3]);  view_1429 = None
    clone_348: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1256, memory_format = torch.contiguous_format);  permute_1256 = None
    view_1430: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_348, [1, 128, 1024]);  clone_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1257: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1427, [0, 2, 1, 3]);  view_1427 = None
    clone_349: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1257, memory_format = torch.contiguous_format);  permute_1257 = None
    view_1431: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_349, [1, 128, 1024]);  clone_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1432: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1431, [128, 1024]);  view_1431 = None
    mm_309: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1432, permute_1258);  permute_1258 = None
    permute_1259: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1432, [1, 0])
    mm_310: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1259, view_123);  permute_1259 = None
    permute_1260: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_310, [1, 0]);  mm_310 = None
    sum_420: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1432, [0], True);  view_1432 = None
    view_1433: "f32[1024]" = torch.ops.aten.reshape.default(sum_420, [1024]);  sum_420 = None
    permute_1261: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1260, [1, 0]);  permute_1260 = None
    view_1434: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_309, [1, 128, 1024]);  mm_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1262: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1428, [0, 2, 1, 3]);  view_1428 = None
    view_1435: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1262, [1, 128, 1024]);  permute_1262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1436: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1435, [128, 1024]);  view_1435 = None
    mm_311: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1436, permute_1263);  permute_1263 = None
    permute_1264: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1436, [1, 0])
    mm_312: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1264, view_123);  permute_1264 = None
    permute_1265: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_312, [1, 0]);  mm_312 = None
    sum_421: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1436, [0], True);  view_1436 = None
    view_1437: "f32[1024]" = torch.ops.aten.reshape.default(sum_421, [1024]);  sum_421 = None
    permute_1266: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1265, [1, 0]);  permute_1265 = None
    view_1438: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_311, [1, 128, 1024]);  mm_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_357: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1434, view_1438);  view_1434 = view_1438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_599: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1430, 0.125);  view_1430 = None
    view_1439: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_599, [128, 1024]);  mul_599 = None
    mm_313: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1439, permute_1267);  permute_1267 = None
    permute_1268: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1439, [1, 0])
    mm_314: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1268, view_123);  permute_1268 = view_123 = None
    permute_1269: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_314, [1, 0]);  mm_314 = None
    sum_422: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1439, [0], True);  view_1439 = None
    view_1440: "f32[1024]" = torch.ops.aten.reshape.default(sum_422, [1024]);  sum_422 = None
    permute_1270: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1269, [1, 0]);  permute_1269 = None
    view_1441: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_313, [1, 128, 1024]);  mm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_358: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_357, view_1441);  add_357 = view_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_601: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_358, primals_98);  primals_98 = None
    mul_602: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_601, 1024)
    sum_423: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True)
    mul_603: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_601, mul_32);  mul_601 = None
    sum_424: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [2], True);  mul_603 = None
    mul_604: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, sum_424);  sum_424 = None
    sub_279: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_602, sum_423);  mul_602 = sum_423 = None
    sub_280: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_279, mul_604);  sub_279 = mul_604 = None
    mul_605: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_87, sub_280);  div_87 = sub_280 = None
    mul_606: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_358, mul_32);  mul_32 = None
    sum_425: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 1]);  mul_606 = None
    sum_426: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_358, [0, 1]);  add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_359: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_356, mul_605);  add_356 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1442: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_359, [128, 1024])
    mm_315: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1442, permute_1271);  permute_1271 = None
    permute_1272: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1442, [1, 0])
    mm_316: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1272, view_121);  permute_1272 = view_121 = None
    permute_1273: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_316, [1, 0]);  mm_316 = None
    sum_427: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1442, [0], True);  view_1442 = None
    view_1443: "f32[1024]" = torch.ops.aten.reshape.default(sum_427, [1024]);  sum_427 = None
    permute_1274: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1273, [1, 0]);  permute_1273 = None
    view_1444: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_24: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_18, full_default_1, view_1444);  le_18 = view_1444 = None
    view_1445: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_24, [128, 4096]);  where_24 = None
    mm_317: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1445, permute_1275);  permute_1275 = None
    permute_1276: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1445, [1, 0])
    mm_318: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1276, view_119);  permute_1276 = view_119 = None
    permute_1277: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_318, [1, 0]);  mm_318 = None
    sum_428: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1445, [0], True);  view_1445 = None
    view_1446: "f32[4096]" = torch.ops.aten.reshape.default(sum_428, [4096]);  sum_428 = None
    permute_1278: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1277, [1, 0]);  permute_1277 = None
    view_1447: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_317, [1, 128, 1024]);  mm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_608: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1447, primals_92);  primals_92 = None
    mul_609: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_608, 1024)
    sum_429: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True)
    mul_610: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_608, mul_30);  mul_608 = None
    sum_430: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [2], True);  mul_610 = None
    mul_611: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_30, sum_430);  sum_430 = None
    sub_282: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_609, sum_429);  mul_609 = sum_429 = None
    sub_283: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_282, mul_611);  sub_282 = mul_611 = None
    mul_612: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_88, sub_283);  div_88 = sub_283 = None
    mul_613: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1447, mul_30);  mul_30 = None
    sum_431: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1]);  mul_613 = None
    sum_432: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1447, [0, 1]);  view_1447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_360: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_359, mul_612);  add_359 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1448: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_360, [128, 1024])
    mm_319: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1448, permute_1279);  permute_1279 = None
    permute_1280: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1448, [1, 0])
    mm_320: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1280, view_117);  permute_1280 = view_117 = None
    permute_1281: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_320, [1, 0]);  mm_320 = None
    sum_433: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1448, [0], True);  view_1448 = None
    view_1449: "f32[1024]" = torch.ops.aten.reshape.default(sum_433, [1024]);  sum_433 = None
    permute_1282: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1281, [1, 0]);  permute_1281 = None
    view_1450: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_319, [1, 128, 1024]);  mm_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1451: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1450, [1, 128, 16, 64]);  view_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1283: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1451, [0, 2, 1, 3]);  view_1451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1452: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1283, [16, 128, 64]);  permute_1283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_192: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1284, view_1452);  permute_1284 = None
    bmm_193: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1452, permute_1285);  view_1452 = permute_1285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_614: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_193, div_5);  bmm_193 = None
    sum_434: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
    mul_615: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_5, sum_434);  div_5 = sum_434 = None
    sub_284: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_194: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1286, sub_284);  permute_1286 = None
    bmm_195: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_284, permute_1287);  sub_284 = permute_1287 = None
    permute_1288: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_194, [0, 2, 1]);  bmm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1453: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_192, [1, 16, 128, 64]);  bmm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1454: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1288, [1, 16, 128, 64]);  permute_1288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1455: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_195, [1, 16, 128, 64]);  bmm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1289: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1455, [0, 2, 1, 3]);  view_1455 = None
    clone_350: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1289, memory_format = torch.contiguous_format);  permute_1289 = None
    view_1456: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_350, [1, 128, 1024]);  clone_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1290: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1453, [0, 2, 1, 3]);  view_1453 = None
    clone_351: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1290, memory_format = torch.contiguous_format);  permute_1290 = None
    view_1457: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_351, [1, 128, 1024]);  clone_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1458: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1457, [128, 1024]);  view_1457 = None
    mm_321: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1458, permute_1291);  permute_1291 = None
    permute_1292: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1458, [1, 0])
    mm_322: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1292, view_103);  permute_1292 = None
    permute_1293: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_322, [1, 0]);  mm_322 = None
    sum_435: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1458, [0], True);  view_1458 = None
    view_1459: "f32[1024]" = torch.ops.aten.reshape.default(sum_435, [1024]);  sum_435 = None
    permute_1294: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1293, [1, 0]);  permute_1293 = None
    view_1460: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_321, [1, 128, 1024]);  mm_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1295: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1454, [0, 2, 1, 3]);  view_1454 = None
    view_1461: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1295, [1, 128, 1024]);  permute_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1462: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1461, [128, 1024]);  view_1461 = None
    mm_323: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1462, permute_1296);  permute_1296 = None
    permute_1297: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1462, [1, 0])
    mm_324: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1297, view_103);  permute_1297 = None
    permute_1298: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_324, [1, 0]);  mm_324 = None
    sum_436: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1462, [0], True);  view_1462 = None
    view_1463: "f32[1024]" = torch.ops.aten.reshape.default(sum_436, [1024]);  sum_436 = None
    permute_1299: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1298, [1, 0]);  permute_1298 = None
    view_1464: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_323, [1, 128, 1024]);  mm_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_361: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1460, view_1464);  view_1460 = view_1464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_616: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1456, 0.125);  view_1456 = None
    view_1465: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_616, [128, 1024]);  mul_616 = None
    mm_325: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1465, permute_1300);  permute_1300 = None
    permute_1301: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1465, [1, 0])
    mm_326: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1301, view_103);  permute_1301 = view_103 = None
    permute_1302: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_326, [1, 0]);  mm_326 = None
    sum_437: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1465, [0], True);  view_1465 = None
    view_1466: "f32[1024]" = torch.ops.aten.reshape.default(sum_437, [1024]);  sum_437 = None
    permute_1303: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1302, [1, 0]);  permute_1302 = None
    view_1467: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_325, [1, 128, 1024]);  mm_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_362: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_361, view_1467);  add_361 = view_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_618: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_362, primals_82);  primals_82 = None
    mul_619: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_618, 1024)
    sum_438: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True)
    mul_620: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_618, mul_27);  mul_618 = None
    sum_439: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
    mul_621: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_27, sum_439);  sum_439 = None
    sub_286: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_619, sum_438);  mul_619 = sum_438 = None
    sub_287: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_286, mul_621);  sub_286 = mul_621 = None
    mul_622: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_89, sub_287);  div_89 = sub_287 = None
    mul_623: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_362, mul_27);  mul_27 = None
    sum_440: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1]);  mul_623 = None
    sum_441: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 1]);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_363: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_360, mul_622);  add_360 = mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1468: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_363, [128, 1024])
    mm_327: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1468, permute_1304);  permute_1304 = None
    permute_1305: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1468, [1, 0])
    mm_328: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1305, view_101);  permute_1305 = view_101 = None
    permute_1306: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_328, [1, 0]);  mm_328 = None
    sum_442: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1468, [0], True);  view_1468 = None
    view_1469: "f32[1024]" = torch.ops.aten.reshape.default(sum_442, [1024]);  sum_442 = None
    permute_1307: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1306, [1, 0]);  permute_1306 = None
    view_1470: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_25: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_19, full_default_1, view_1470);  le_19 = view_1470 = None
    view_1471: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_25, [128, 4096]);  where_25 = None
    mm_329: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1471, permute_1308);  permute_1308 = None
    permute_1309: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1471, [1, 0])
    mm_330: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1309, view_99);  permute_1309 = view_99 = None
    permute_1310: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_330, [1, 0]);  mm_330 = None
    sum_443: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1471, [0], True);  view_1471 = None
    view_1472: "f32[4096]" = torch.ops.aten.reshape.default(sum_443, [4096]);  sum_443 = None
    permute_1311: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1310, [1, 0]);  permute_1310 = None
    view_1473: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_329, [1, 128, 1024]);  mm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_625: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1473, primals_76);  primals_76 = None
    mul_626: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_625, 1024)
    sum_444: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True)
    mul_627: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_625, mul_25);  mul_625 = None
    sum_445: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True);  mul_627 = None
    mul_628: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, sum_445);  sum_445 = None
    sub_289: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_626, sum_444);  mul_626 = sum_444 = None
    sub_290: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_289, mul_628);  sub_289 = mul_628 = None
    mul_629: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_90, sub_290);  div_90 = sub_290 = None
    mul_630: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1473, mul_25);  mul_25 = None
    sum_446: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 1]);  mul_630 = None
    sum_447: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1473, [0, 1]);  view_1473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_364: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_363, mul_629);  add_363 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1474: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_364, [128, 1024])
    mm_331: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1474, permute_1312);  permute_1312 = None
    permute_1313: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1474, [1, 0])
    mm_332: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1313, view_97);  permute_1313 = view_97 = None
    permute_1314: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_332, [1, 0]);  mm_332 = None
    sum_448: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1474, [0], True);  view_1474 = None
    view_1475: "f32[1024]" = torch.ops.aten.reshape.default(sum_448, [1024]);  sum_448 = None
    permute_1315: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1314, [1, 0]);  permute_1314 = None
    view_1476: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_331, [1, 128, 1024]);  mm_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1477: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1476, [1, 128, 16, 64]);  view_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1316: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1477, [0, 2, 1, 3]);  view_1477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1478: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1316, [16, 128, 64]);  permute_1316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_196: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1317, view_1478);  permute_1317 = None
    bmm_197: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1478, permute_1318);  view_1478 = permute_1318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_631: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_197, div_4);  bmm_197 = None
    sum_449: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_631, [-1], True)
    mul_632: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_4, sum_449);  div_4 = sum_449 = None
    sub_291: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_198: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1319, sub_291);  permute_1319 = None
    bmm_199: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_291, permute_1320);  sub_291 = permute_1320 = None
    permute_1321: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_198, [0, 2, 1]);  bmm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1479: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_196, [1, 16, 128, 64]);  bmm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1480: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1321, [1, 16, 128, 64]);  permute_1321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1481: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_199, [1, 16, 128, 64]);  bmm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1322: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1481, [0, 2, 1, 3]);  view_1481 = None
    clone_352: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1322, memory_format = torch.contiguous_format);  permute_1322 = None
    view_1482: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_352, [1, 128, 1024]);  clone_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1323: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1479, [0, 2, 1, 3]);  view_1479 = None
    clone_353: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1323, memory_format = torch.contiguous_format);  permute_1323 = None
    view_1483: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_353, [1, 128, 1024]);  clone_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1484: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1483, [128, 1024]);  view_1483 = None
    mm_333: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1484, permute_1324);  permute_1324 = None
    permute_1325: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1484, [1, 0])
    mm_334: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1325, view_83);  permute_1325 = None
    permute_1326: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_334, [1, 0]);  mm_334 = None
    sum_450: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1484, [0], True);  view_1484 = None
    view_1485: "f32[1024]" = torch.ops.aten.reshape.default(sum_450, [1024]);  sum_450 = None
    permute_1327: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1326, [1, 0]);  permute_1326 = None
    view_1486: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_333, [1, 128, 1024]);  mm_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1328: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1480, [0, 2, 1, 3]);  view_1480 = None
    view_1487: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1328, [1, 128, 1024]);  permute_1328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1488: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1487, [128, 1024]);  view_1487 = None
    mm_335: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1488, permute_1329);  permute_1329 = None
    permute_1330: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1488, [1, 0])
    mm_336: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1330, view_83);  permute_1330 = None
    permute_1331: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_336, [1, 0]);  mm_336 = None
    sum_451: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1488, [0], True);  view_1488 = None
    view_1489: "f32[1024]" = torch.ops.aten.reshape.default(sum_451, [1024]);  sum_451 = None
    permute_1332: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1331, [1, 0]);  permute_1331 = None
    view_1490: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_335, [1, 128, 1024]);  mm_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_365: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1486, view_1490);  view_1486 = view_1490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_633: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1482, 0.125);  view_1482 = None
    view_1491: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_633, [128, 1024]);  mul_633 = None
    mm_337: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1491, permute_1333);  permute_1333 = None
    permute_1334: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1491, [1, 0])
    mm_338: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1334, view_83);  permute_1334 = view_83 = None
    permute_1335: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_338, [1, 0]);  mm_338 = None
    sum_452: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1491, [0], True);  view_1491 = None
    view_1492: "f32[1024]" = torch.ops.aten.reshape.default(sum_452, [1024]);  sum_452 = None
    permute_1336: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1335, [1, 0]);  permute_1335 = None
    view_1493: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_337, [1, 128, 1024]);  mm_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_366: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_365, view_1493);  add_365 = view_1493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_635: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_366, primals_66);  primals_66 = None
    mul_636: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_635, 1024)
    sum_453: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_635, mul_22);  mul_635 = None
    sum_454: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_22, sum_454);  sum_454 = None
    sub_293: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_636, sum_453);  mul_636 = sum_453 = None
    sub_294: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_293, mul_638);  sub_293 = mul_638 = None
    mul_639: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_91, sub_294);  div_91 = sub_294 = None
    mul_640: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_366, mul_22);  mul_22 = None
    sum_455: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_456: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_366, [0, 1]);  add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_367: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_364, mul_639);  add_364 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1494: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_367, [128, 1024])
    mm_339: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1494, permute_1337);  permute_1337 = None
    permute_1338: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1494, [1, 0])
    mm_340: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1338, view_81);  permute_1338 = view_81 = None
    permute_1339: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_340, [1, 0]);  mm_340 = None
    sum_457: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1494, [0], True);  view_1494 = None
    view_1495: "f32[1024]" = torch.ops.aten.reshape.default(sum_457, [1024]);  sum_457 = None
    permute_1340: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1339, [1, 0]);  permute_1339 = None
    view_1496: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_26: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_20, full_default_1, view_1496);  le_20 = view_1496 = None
    view_1497: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_26, [128, 4096]);  where_26 = None
    mm_341: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1497, permute_1341);  permute_1341 = None
    permute_1342: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1497, [1, 0])
    mm_342: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1342, view_79);  permute_1342 = view_79 = None
    permute_1343: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_342, [1, 0]);  mm_342 = None
    sum_458: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1497, [0], True);  view_1497 = None
    view_1498: "f32[4096]" = torch.ops.aten.reshape.default(sum_458, [4096]);  sum_458 = None
    permute_1344: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1343, [1, 0]);  permute_1343 = None
    view_1499: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_341, [1, 128, 1024]);  mm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_642: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1499, primals_60);  primals_60 = None
    mul_643: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_642, 1024)
    sum_459: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
    mul_644: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_642, mul_20);  mul_642 = None
    sum_460: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
    mul_645: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, sum_460);  sum_460 = None
    sub_296: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_643, sum_459);  mul_643 = sum_459 = None
    sub_297: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_296, mul_645);  sub_296 = mul_645 = None
    mul_646: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_92, sub_297);  div_92 = sub_297 = None
    mul_647: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1499, mul_20);  mul_20 = None
    sum_461: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
    sum_462: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1499, [0, 1]);  view_1499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_368: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_367, mul_646);  add_367 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1500: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_368, [128, 1024])
    mm_343: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1500, permute_1345);  permute_1345 = None
    permute_1346: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1500, [1, 0])
    mm_344: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1346, view_77);  permute_1346 = view_77 = None
    permute_1347: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_344, [1, 0]);  mm_344 = None
    sum_463: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1500, [0], True);  view_1500 = None
    view_1501: "f32[1024]" = torch.ops.aten.reshape.default(sum_463, [1024]);  sum_463 = None
    permute_1348: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1347, [1, 0]);  permute_1347 = None
    view_1502: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_343, [1, 128, 1024]);  mm_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1503: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1502, [1, 128, 16, 64]);  view_1502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1349: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1503, [0, 2, 1, 3]);  view_1503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1504: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1349, [16, 128, 64]);  permute_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_200: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1350, view_1504);  permute_1350 = None
    bmm_201: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1504, permute_1351);  view_1504 = permute_1351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_648: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_201, div_3);  bmm_201 = None
    sum_464: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_648, [-1], True)
    mul_649: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_3, sum_464);  div_3 = sum_464 = None
    sub_298: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_202: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1352, sub_298);  permute_1352 = None
    bmm_203: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_298, permute_1353);  sub_298 = permute_1353 = None
    permute_1354: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_202, [0, 2, 1]);  bmm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1505: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_200, [1, 16, 128, 64]);  bmm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1506: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1354, [1, 16, 128, 64]);  permute_1354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1507: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_203, [1, 16, 128, 64]);  bmm_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1355: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1507, [0, 2, 1, 3]);  view_1507 = None
    clone_354: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1355, memory_format = torch.contiguous_format);  permute_1355 = None
    view_1508: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_354, [1, 128, 1024]);  clone_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1356: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1505, [0, 2, 1, 3]);  view_1505 = None
    clone_355: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1356, memory_format = torch.contiguous_format);  permute_1356 = None
    view_1509: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_355, [1, 128, 1024]);  clone_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1510: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1509, [128, 1024]);  view_1509 = None
    mm_345: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1510, permute_1357);  permute_1357 = None
    permute_1358: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1510, [1, 0])
    mm_346: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1358, view_63);  permute_1358 = None
    permute_1359: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_346, [1, 0]);  mm_346 = None
    sum_465: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1510, [0], True);  view_1510 = None
    view_1511: "f32[1024]" = torch.ops.aten.reshape.default(sum_465, [1024]);  sum_465 = None
    permute_1360: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1359, [1, 0]);  permute_1359 = None
    view_1512: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_345, [1, 128, 1024]);  mm_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1361: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1506, [0, 2, 1, 3]);  view_1506 = None
    view_1513: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1361, [1, 128, 1024]);  permute_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1514: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1513, [128, 1024]);  view_1513 = None
    mm_347: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1514, permute_1362);  permute_1362 = None
    permute_1363: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1514, [1, 0])
    mm_348: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1363, view_63);  permute_1363 = None
    permute_1364: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_348, [1, 0]);  mm_348 = None
    sum_466: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1514, [0], True);  view_1514 = None
    view_1515: "f32[1024]" = torch.ops.aten.reshape.default(sum_466, [1024]);  sum_466 = None
    permute_1365: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1364, [1, 0]);  permute_1364 = None
    view_1516: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_347, [1, 128, 1024]);  mm_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_369: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1512, view_1516);  view_1512 = view_1516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_650: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1508, 0.125);  view_1508 = None
    view_1517: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_650, [128, 1024]);  mul_650 = None
    mm_349: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1517, permute_1366);  permute_1366 = None
    permute_1367: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1517, [1, 0])
    mm_350: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1367, view_63);  permute_1367 = view_63 = None
    permute_1368: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_350, [1, 0]);  mm_350 = None
    sum_467: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1517, [0], True);  view_1517 = None
    view_1518: "f32[1024]" = torch.ops.aten.reshape.default(sum_467, [1024]);  sum_467 = None
    permute_1369: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1368, [1, 0]);  permute_1368 = None
    view_1519: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_349, [1, 128, 1024]);  mm_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_370: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_369, view_1519);  add_369 = view_1519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_652: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_370, primals_50);  primals_50 = None
    mul_653: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_652, 1024)
    sum_468: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [2], True)
    mul_654: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_652, mul_17);  mul_652 = None
    sum_469: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_654, [2], True);  mul_654 = None
    mul_655: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, sum_469);  sum_469 = None
    sub_300: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_653, sum_468);  mul_653 = sum_468 = None
    sub_301: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_300, mul_655);  sub_300 = mul_655 = None
    mul_656: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_93, sub_301);  div_93 = sub_301 = None
    mul_657: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_370, mul_17);  mul_17 = None
    sum_470: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 1]);  mul_657 = None
    sum_471: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 1]);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_371: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_368, mul_656);  add_368 = mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1520: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_371, [128, 1024])
    mm_351: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1520, permute_1370);  permute_1370 = None
    permute_1371: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1520, [1, 0])
    mm_352: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1371, view_61);  permute_1371 = view_61 = None
    permute_1372: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_352, [1, 0]);  mm_352 = None
    sum_472: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1520, [0], True);  view_1520 = None
    view_1521: "f32[1024]" = torch.ops.aten.reshape.default(sum_472, [1024]);  sum_472 = None
    permute_1373: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1372, [1, 0]);  permute_1372 = None
    view_1522: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_27: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_21, full_default_1, view_1522);  le_21 = view_1522 = None
    view_1523: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_27, [128, 4096]);  where_27 = None
    mm_353: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1523, permute_1374);  permute_1374 = None
    permute_1375: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1523, [1, 0])
    mm_354: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1375, view_59);  permute_1375 = view_59 = None
    permute_1376: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_354, [1, 0]);  mm_354 = None
    sum_473: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1523, [0], True);  view_1523 = None
    view_1524: "f32[4096]" = torch.ops.aten.reshape.default(sum_473, [4096]);  sum_473 = None
    permute_1377: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1376, [1, 0]);  permute_1376 = None
    view_1525: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_353, [1, 128, 1024]);  mm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_659: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1525, primals_44);  primals_44 = None
    mul_660: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_659, 1024)
    sum_474: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True)
    mul_661: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_659, mul_15);  mul_659 = None
    sum_475: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True);  mul_661 = None
    mul_662: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, sum_475);  sum_475 = None
    sub_303: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_660, sum_474);  mul_660 = sum_474 = None
    sub_304: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_303, mul_662);  sub_303 = mul_662 = None
    mul_663: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_94, sub_304);  div_94 = sub_304 = None
    mul_664: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1525, mul_15);  mul_15 = None
    sum_476: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_664, [0, 1]);  mul_664 = None
    sum_477: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1525, [0, 1]);  view_1525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_372: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_371, mul_663);  add_371 = mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1526: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_372, [128, 1024])
    mm_355: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1526, permute_1378);  permute_1378 = None
    permute_1379: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1526, [1, 0])
    mm_356: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1379, view_57);  permute_1379 = view_57 = None
    permute_1380: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_356, [1, 0]);  mm_356 = None
    sum_478: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1526, [0], True);  view_1526 = None
    view_1527: "f32[1024]" = torch.ops.aten.reshape.default(sum_478, [1024]);  sum_478 = None
    permute_1381: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1380, [1, 0]);  permute_1380 = None
    view_1528: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_355, [1, 128, 1024]);  mm_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1529: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1528, [1, 128, 16, 64]);  view_1528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1382: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1529, [0, 2, 1, 3]);  view_1529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1530: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1382, [16, 128, 64]);  permute_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_204: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1383, view_1530);  permute_1383 = None
    bmm_205: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1530, permute_1384);  view_1530 = permute_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_665: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_205, div_2);  bmm_205 = None
    sum_479: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [-1], True)
    mul_666: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_2, sum_479);  div_2 = sum_479 = None
    sub_305: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_206: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1385, sub_305);  permute_1385 = None
    bmm_207: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_305, permute_1386);  sub_305 = permute_1386 = None
    permute_1387: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_206, [0, 2, 1]);  bmm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1531: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_204, [1, 16, 128, 64]);  bmm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1532: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1387, [1, 16, 128, 64]);  permute_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1533: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_207, [1, 16, 128, 64]);  bmm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1388: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1533, [0, 2, 1, 3]);  view_1533 = None
    clone_356: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1388, memory_format = torch.contiguous_format);  permute_1388 = None
    view_1534: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_356, [1, 128, 1024]);  clone_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1389: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1531, [0, 2, 1, 3]);  view_1531 = None
    clone_357: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1389, memory_format = torch.contiguous_format);  permute_1389 = None
    view_1535: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_357, [1, 128, 1024]);  clone_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1536: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1535, [128, 1024]);  view_1535 = None
    mm_357: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1536, permute_1390);  permute_1390 = None
    permute_1391: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1536, [1, 0])
    mm_358: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1391, view_43);  permute_1391 = None
    permute_1392: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_358, [1, 0]);  mm_358 = None
    sum_480: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1536, [0], True);  view_1536 = None
    view_1537: "f32[1024]" = torch.ops.aten.reshape.default(sum_480, [1024]);  sum_480 = None
    permute_1393: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1392, [1, 0]);  permute_1392 = None
    view_1538: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_357, [1, 128, 1024]);  mm_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1394: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1532, [0, 2, 1, 3]);  view_1532 = None
    view_1539: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1394, [1, 128, 1024]);  permute_1394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1540: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1539, [128, 1024]);  view_1539 = None
    mm_359: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1540, permute_1395);  permute_1395 = None
    permute_1396: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1540, [1, 0])
    mm_360: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1396, view_43);  permute_1396 = None
    permute_1397: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_360, [1, 0]);  mm_360 = None
    sum_481: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1540, [0], True);  view_1540 = None
    view_1541: "f32[1024]" = torch.ops.aten.reshape.default(sum_481, [1024]);  sum_481 = None
    permute_1398: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1397, [1, 0]);  permute_1397 = None
    view_1542: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_359, [1, 128, 1024]);  mm_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_373: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1538, view_1542);  view_1538 = view_1542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_667: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1534, 0.125);  view_1534 = None
    view_1543: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_667, [128, 1024]);  mul_667 = None
    mm_361: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1543, permute_1399);  permute_1399 = None
    permute_1400: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1543, [1, 0])
    mm_362: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1400, view_43);  permute_1400 = view_43 = None
    permute_1401: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_362, [1, 0]);  mm_362 = None
    sum_482: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1543, [0], True);  view_1543 = None
    view_1544: "f32[1024]" = torch.ops.aten.reshape.default(sum_482, [1024]);  sum_482 = None
    permute_1402: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1401, [1, 0]);  permute_1401 = None
    view_1545: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_361, [1, 128, 1024]);  mm_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_374: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_373, view_1545);  add_373 = view_1545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_669: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_374, primals_34);  primals_34 = None
    mul_670: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_669, 1024)
    sum_483: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2], True)
    mul_671: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_669, mul_12);  mul_669 = None
    sum_484: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
    mul_672: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, sum_484);  sum_484 = None
    sub_307: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_670, sum_483);  mul_670 = sum_483 = None
    sub_308: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_307, mul_672);  sub_307 = mul_672 = None
    mul_673: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_95, sub_308);  div_95 = sub_308 = None
    mul_674: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_374, mul_12);  mul_12 = None
    sum_485: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 1]);  mul_674 = None
    sum_486: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_374, [0, 1]);  add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_375: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_372, mul_673);  add_372 = mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1546: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_375, [128, 1024])
    mm_363: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1546, permute_1403);  permute_1403 = None
    permute_1404: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1546, [1, 0])
    mm_364: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1404, view_41);  permute_1404 = view_41 = None
    permute_1405: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_364, [1, 0]);  mm_364 = None
    sum_487: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1546, [0], True);  view_1546 = None
    view_1547: "f32[1024]" = torch.ops.aten.reshape.default(sum_487, [1024]);  sum_487 = None
    permute_1406: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1405, [1, 0]);  permute_1405 = None
    view_1548: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_28: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_22, full_default_1, view_1548);  le_22 = view_1548 = None
    view_1549: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_28, [128, 4096]);  where_28 = None
    mm_365: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1549, permute_1407);  permute_1407 = None
    permute_1408: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1549, [1, 0])
    mm_366: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1408, view_39);  permute_1408 = view_39 = None
    permute_1409: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_366, [1, 0]);  mm_366 = None
    sum_488: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1549, [0], True);  view_1549 = None
    view_1550: "f32[4096]" = torch.ops.aten.reshape.default(sum_488, [4096]);  sum_488 = None
    permute_1410: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1409, [1, 0]);  permute_1409 = None
    view_1551: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_365, [1, 128, 1024]);  mm_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_676: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1551, primals_28);  primals_28 = None
    mul_677: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_676, 1024)
    sum_489: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_676, [2], True)
    mul_678: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_676, mul_10);  mul_676 = None
    sum_490: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_678, [2], True);  mul_678 = None
    mul_679: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, sum_490);  sum_490 = None
    sub_310: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_677, sum_489);  mul_677 = sum_489 = None
    sub_311: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_310, mul_679);  sub_310 = mul_679 = None
    mul_680: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_96, sub_311);  div_96 = sub_311 = None
    mul_681: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1551, mul_10);  mul_10 = None
    sum_491: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 1]);  mul_681 = None
    sum_492: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1551, [0, 1]);  view_1551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_376: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_375, mul_680);  add_375 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1552: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_376, [128, 1024])
    mm_367: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1552, permute_1411);  permute_1411 = None
    permute_1412: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1552, [1, 0])
    mm_368: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1412, view_37);  permute_1412 = view_37 = None
    permute_1413: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_368, [1, 0]);  mm_368 = None
    sum_493: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1552, [0], True);  view_1552 = None
    view_1553: "f32[1024]" = torch.ops.aten.reshape.default(sum_493, [1024]);  sum_493 = None
    permute_1414: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1413, [1, 0]);  permute_1413 = None
    view_1554: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_367, [1, 128, 1024]);  mm_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1555: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1554, [1, 128, 16, 64]);  view_1554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1415: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1555, [0, 2, 1, 3]);  view_1555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1556: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1415, [16, 128, 64]);  permute_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_208: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1416, view_1556);  permute_1416 = None
    bmm_209: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1556, permute_1417);  view_1556 = permute_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_682: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_209, div_1);  bmm_209 = None
    sum_494: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [-1], True)
    mul_683: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div_1, sum_494);  div_1 = sum_494 = None
    sub_312: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_210: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1418, sub_312);  permute_1418 = None
    bmm_211: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_312, permute_1419);  sub_312 = permute_1419 = None
    permute_1420: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_210, [0, 2, 1]);  bmm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1557: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_208, [1, 16, 128, 64]);  bmm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1558: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1420, [1, 16, 128, 64]);  permute_1420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1559: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_211, [1, 16, 128, 64]);  bmm_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1421: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1559, [0, 2, 1, 3]);  view_1559 = None
    clone_358: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1421, memory_format = torch.contiguous_format);  permute_1421 = None
    view_1560: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_358, [1, 128, 1024]);  clone_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1422: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1557, [0, 2, 1, 3]);  view_1557 = None
    clone_359: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1422, memory_format = torch.contiguous_format);  permute_1422 = None
    view_1561: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_359, [1, 128, 1024]);  clone_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1562: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1561, [128, 1024]);  view_1561 = None
    mm_369: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1562, permute_1423);  permute_1423 = None
    permute_1424: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1562, [1, 0])
    mm_370: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1424, view_23);  permute_1424 = None
    permute_1425: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_370, [1, 0]);  mm_370 = None
    sum_495: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1562, [0], True);  view_1562 = None
    view_1563: "f32[1024]" = torch.ops.aten.reshape.default(sum_495, [1024]);  sum_495 = None
    permute_1426: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1425, [1, 0]);  permute_1425 = None
    view_1564: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_369, [1, 128, 1024]);  mm_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1427: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1558, [0, 2, 1, 3]);  view_1558 = None
    view_1565: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1427, [1, 128, 1024]);  permute_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1566: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1565, [128, 1024]);  view_1565 = None
    mm_371: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1566, permute_1428);  permute_1428 = None
    permute_1429: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1566, [1, 0])
    mm_372: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1429, view_23);  permute_1429 = None
    permute_1430: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_372, [1, 0]);  mm_372 = None
    sum_496: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1566, [0], True);  view_1566 = None
    view_1567: "f32[1024]" = torch.ops.aten.reshape.default(sum_496, [1024]);  sum_496 = None
    permute_1431: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1430, [1, 0]);  permute_1430 = None
    view_1568: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_371, [1, 128, 1024]);  mm_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_377: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1564, view_1568);  view_1564 = view_1568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_684: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1560, 0.125);  view_1560 = None
    view_1569: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_684, [128, 1024]);  mul_684 = None
    mm_373: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1569, permute_1432);  permute_1432 = None
    permute_1433: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1569, [1, 0])
    mm_374: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1433, view_23);  permute_1433 = view_23 = None
    permute_1434: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_374, [1, 0]);  mm_374 = None
    sum_497: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1569, [0], True);  view_1569 = None
    view_1570: "f32[1024]" = torch.ops.aten.reshape.default(sum_497, [1024]);  sum_497 = None
    permute_1435: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1434, [1, 0]);  permute_1434 = None
    view_1571: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_373, [1, 128, 1024]);  mm_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_378: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_377, view_1571);  add_377 = view_1571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_686: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_378, primals_18);  primals_18 = None
    mul_687: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_686, 1024)
    sum_498: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
    mul_688: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_686, mul_7);  mul_686 = None
    sum_499: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
    mul_689: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, sum_499);  sum_499 = None
    sub_314: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_687, sum_498);  mul_687 = sum_498 = None
    sub_315: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_314, mul_689);  sub_314 = mul_689 = None
    mul_690: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_97, sub_315);  div_97 = sub_315 = None
    mul_691: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_378, mul_7);  mul_7 = None
    sum_500: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
    sum_501: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_379: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_376, mul_690);  add_376 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:402, code: hidden_states = self.fc2(hidden_states)
    view_1572: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_379, [128, 1024])
    mm_375: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1572, permute_1436);  permute_1436 = None
    permute_1437: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1572, [1, 0])
    mm_376: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1437, view_21);  permute_1437 = view_21 = None
    permute_1438: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_376, [1, 0]);  mm_376 = None
    sum_502: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1572, [0], True);  view_1572 = None
    view_1573: "f32[1024]" = torch.ops.aten.reshape.default(sum_502, [1024]);  sum_502 = None
    permute_1439: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1438, [1, 0]);  permute_1438 = None
    view_1574: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:400, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    where_29: "f32[1, 128, 4096]" = torch.ops.aten.where.self(le_23, full_default_1, view_1574);  le_23 = view_1574 = None
    view_1575: "f32[128, 4096]" = torch.ops.aten.reshape.default(where_29, [128, 4096]);  where_29 = None
    mm_377: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1575, permute_1440);  permute_1440 = None
    permute_1441: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1575, [1, 0])
    mm_378: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1441, view_19);  permute_1441 = view_19 = None
    permute_1442: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_378, [1, 0]);  mm_378 = None
    sum_503: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1575, [0], True);  view_1575 = None
    view_1576: "f32[4096]" = torch.ops.aten.reshape.default(sum_503, [4096]);  sum_503 = None
    permute_1443: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1442, [1, 0]);  permute_1442 = None
    view_1577: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_377, [1, 128, 1024]);  mm_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    mul_693: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1577, primals_12);  primals_12 = None
    mul_694: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_693, 1024)
    sum_504: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True)
    mul_695: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_693, mul_5);  mul_693 = None
    sum_505: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_695, [2], True);  mul_695 = None
    mul_696: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_5, sum_505);  sum_505 = None
    sub_317: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_694, sum_504);  mul_694 = sum_504 = None
    sub_318: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_317, mul_696);  sub_317 = mul_696 = None
    mul_697: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_98, sub_318);  div_98 = sub_318 = None
    mul_698: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1577, mul_5);  mul_5 = None
    sum_506: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 1]);  mul_698 = None
    sum_507: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1577, [0, 1]);  view_1577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:399, code: hidden_states = self.final_layer_norm(hidden_states)
    add_380: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_379, mul_697);  add_379 = mul_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:346, code: attn_output = self.out_proj(attn_output)
    view_1578: "f32[128, 1024]" = torch.ops.aten.reshape.default(add_380, [128, 1024])
    mm_379: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1578, permute_1444);  permute_1444 = None
    permute_1445: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1578, [1, 0])
    mm_380: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1445, view_17);  permute_1445 = view_17 = None
    permute_1446: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_380, [1, 0]);  mm_380 = None
    sum_508: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1578, [0], True);  view_1578 = None
    view_1579: "f32[1024]" = torch.ops.aten.reshape.default(sum_508, [1024]);  sum_508 = None
    permute_1447: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1446, [1, 0]);  permute_1446 = None
    view_1580: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_379, [1, 128, 1024]);  mm_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:344, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    view_1581: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1580, [1, 128, 16, 64]);  view_1580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:340, code: attn_output = attn_output.transpose(1, 2)
    permute_1448: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_1581, [0, 2, 1, 3]);  view_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:339, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_1582: "f32[16, 128, 64]" = torch.ops.aten.reshape.default(permute_1448, [16, 128, 64]);  permute_1448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:331, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_212: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(permute_1449, view_1582);  permute_1449 = None
    bmm_213: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1582, permute_1450);  view_1582 = permute_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:308, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_699: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_213, div);  bmm_213 = None
    sum_509: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [-1], True)
    mul_700: "f32[16, 128, 128]" = torch.ops.aten.mul.Tensor(div, sum_509);  div = sum_509 = None
    sub_319: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:292, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    bmm_214: "f32[16, 64, 128]" = torch.ops.aten.bmm.default(permute_1451, sub_319);  permute_1451 = None
    bmm_215: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(sub_319, permute_1452);  sub_319 = permute_1452 = None
    permute_1453: "f32[16, 128, 64]" = torch.ops.aten.permute.default(bmm_214, [0, 2, 1]);  bmm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:289, code: value_states = value_states.reshape(*proj_shape)
    view_1583: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_212, [1, 16, 128, 64]);  bmm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:288, code: key_states = key_states.reshape(*proj_shape)
    view_1584: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_1453, [1, 16, 128, 64]);  permute_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:287, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_1585: "f32[1, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_215, [1, 16, 128, 64]);  bmm_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1454: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1585, [0, 2, 1, 3]);  view_1585 = None
    clone_360: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1454, memory_format = torch.contiguous_format);  permute_1454 = None
    view_1586: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_360, [1, 128, 1024]);  clone_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1455: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1583, [0, 2, 1, 3]);  view_1583 = None
    clone_361: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1455, memory_format = torch.contiguous_format);  permute_1455 = None
    view_1587: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(clone_361, [1, 128, 1024]);  clone_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:274, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_1588: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1587, [128, 1024]);  view_1587 = None
    mm_381: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1588, permute_1456);  permute_1456 = None
    permute_1457: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1588, [1, 0])
    mm_382: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1457, view_3);  permute_1457 = None
    permute_1458: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_382, [1, 0]);  mm_382 = None
    sum_510: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1588, [0], True);  view_1588 = None
    view_1589: "f32[1024]" = torch.ops.aten.reshape.default(sum_510, [1024]);  sum_510 = None
    permute_1459: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1458, [1, 0]);  permute_1458 = None
    view_1590: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_381, [1, 128, 1024]);  mm_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:228, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    permute_1460: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_1584, [0, 2, 1, 3]);  view_1584 = None
    view_1591: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(permute_1460, [1, 128, 1024]);  permute_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_1592: "f32[128, 1024]" = torch.ops.aten.reshape.default(view_1591, [128, 1024]);  view_1591 = None
    mm_383: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1592, permute_1461);  permute_1461 = None
    permute_1462: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1592, [1, 0])
    mm_384: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1462, view_3);  permute_1462 = None
    permute_1463: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_384, [1, 0]);  mm_384 = None
    sum_511: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1592, [0], True);  view_1592 = None
    view_1593: "f32[1024]" = torch.ops.aten.reshape.default(sum_511, [1024]);  sum_511 = None
    permute_1464: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1463, [1, 0]);  permute_1463 = None
    view_1594: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_383, [1, 128, 1024]);  mm_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:273, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    add_381: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_1590, view_1594);  view_1590 = view_1594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    mul_701: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1586, 0.125);  view_1586 = None
    view_1595: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_701, [128, 1024]);  mul_701 = None
    mm_385: "f32[128, 1024]" = torch.ops.aten.mm.default(view_1595, permute_1465);  permute_1465 = None
    permute_1466: "f32[1024, 128]" = torch.ops.aten.permute.default(view_1595, [1, 0])
    mm_386: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1466, view_3);  permute_1466 = view_3 = None
    permute_1467: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_386, [1, 0]);  mm_386 = None
    sum_512: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1595, [0], True);  view_1595 = None
    view_1596: "f32[1024]" = torch.ops.aten.reshape.default(sum_512, [1024]);  sum_512 = None
    permute_1468: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1467, [1, 0]);  permute_1467 = None
    view_1597: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_385, [1, 128, 1024]);  mm_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:248, code: query_states = self.q_proj(hidden_states) * self.scaling
    add_382: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_381, view_1597);  add_381 = view_1597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    mul_703: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_382, primals_2);  primals_2 = None
    mul_704: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_703, 1024)
    sum_513: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
    mul_705: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_703, mul_2);  mul_703 = None
    sum_514: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
    mul_706: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, sum_514);  sum_514 = None
    sub_321: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_704, sum_513);  mul_704 = sum_513 = None
    sub_322: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_321, mul_706);  sub_321 = mul_706 = None
    mul_707: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(div_99, sub_322);  div_99 = sub_322 = None
    mul_708: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_382, mul_2);  mul_2 = None
    sum_515: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
    sum_516: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_382, [0, 1]);  add_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:388, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    add_383: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_380, mul_707);  add_380 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:786, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    mul_709: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_383, 32.0);  add_383 = None
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, 1)
    unsqueeze_7: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_30: "f32[1, 128, 1024]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_709);  unsqueeze_7 = full_default_1 = mul_709 = None
    _unsafe_index_put_1: "f32[128112, 1024]" = torch.ops.prims._unsafe_index_put_.default(full_default_20, [view], where_30, True);  full_default_20 = view = where_30 = None
    return [_unsafe_index_put_1, sum_515, sum_516, permute_1468, view_1596, permute_1464, view_1593, permute_1459, view_1589, permute_1447, view_1579, sum_506, sum_507, permute_1443, view_1576, permute_1439, view_1573, sum_500, sum_501, permute_1435, view_1570, permute_1431, view_1567, permute_1426, view_1563, permute_1414, view_1553, sum_491, sum_492, permute_1410, view_1550, permute_1406, view_1547, sum_485, sum_486, permute_1402, view_1544, permute_1398, view_1541, permute_1393, view_1537, permute_1381, view_1527, sum_476, sum_477, permute_1377, view_1524, permute_1373, view_1521, sum_470, sum_471, permute_1369, view_1518, permute_1365, view_1515, permute_1360, view_1511, permute_1348, view_1501, sum_461, sum_462, permute_1344, view_1498, permute_1340, view_1495, sum_455, sum_456, permute_1336, view_1492, permute_1332, view_1489, permute_1327, view_1485, permute_1315, view_1475, sum_446, sum_447, permute_1311, view_1472, permute_1307, view_1469, sum_440, sum_441, permute_1303, view_1466, permute_1299, view_1463, permute_1294, view_1459, permute_1282, view_1449, sum_431, sum_432, permute_1278, view_1446, permute_1274, view_1443, sum_425, sum_426, permute_1270, view_1440, permute_1266, view_1437, permute_1261, view_1433, permute_1249, view_1423, sum_416, sum_417, permute_1245, view_1420, permute_1241, view_1417, sum_410, sum_411, permute_1237, view_1414, permute_1233, view_1411, permute_1228, view_1407, permute_1216, view_1397, sum_401, sum_402, permute_1212, view_1394, permute_1208, view_1391, sum_395, sum_396, permute_1204, view_1388, permute_1200, view_1385, permute_1195, view_1381, permute_1183, view_1371, sum_386, sum_387, permute_1179, view_1368, permute_1175, view_1365, sum_380, sum_381, permute_1171, view_1362, permute_1167, view_1359, permute_1162, view_1355, permute_1150, view_1345, sum_371, sum_372, permute_1146, view_1342, permute_1142, view_1339, sum_365, sum_366, permute_1138, view_1336, permute_1134, view_1333, permute_1129, view_1329, permute_1117, view_1319, sum_356, sum_357, permute_1113, view_1316, permute_1109, view_1313, sum_350, sum_351, permute_1105, view_1310, permute_1101, view_1307, permute_1096, view_1303, permute_1084, view_1293, sum_341, sum_342, permute_1080, view_1290, permute_1076, view_1287, sum_335, sum_336, _unsafe_index_put, sum_331, sum_332, permute_1072, view_1284, permute_1068, view_1281, permute_1063, view_1277, permute_1051, view_1265, sum_322, sum_323, permute_1047, view_1262, permute_1043, view_1259, permute_1038, view_1255, permute_1026, view_1245, sum_313, sum_314, permute_1022, view_1242, permute_1018, view_1239, sum_307, sum_308, permute_1014, view_1236, permute_1010, view_1233, permute_1005, view_1229, permute_993, view_1217, sum_298, sum_299, permute_989, view_1214, permute_985, view_1211, permute_980, view_1207, permute_968, view_1197, sum_289, sum_290, permute_964, view_1194, permute_960, view_1191, sum_283, sum_284, permute_956, view_1188, permute_952, view_1185, permute_947, view_1181, permute_935, view_1169, sum_274, sum_275, permute_931, view_1166, permute_927, view_1163, permute_922, view_1159, permute_910, view_1149, sum_265, sum_266, permute_906, view_1146, permute_902, view_1143, sum_259, sum_260, permute_898, view_1140, permute_894, view_1137, permute_889, view_1133, permute_877, view_1121, sum_250, sum_251, permute_873, view_1118, permute_869, view_1115, permute_864, view_1111, permute_852, view_1101, sum_241, sum_242, permute_848, view_1098, permute_844, view_1095, sum_235, sum_236, permute_840, view_1092, permute_836, view_1089, permute_831, view_1085, permute_819, view_1073, sum_226, sum_227, permute_815, view_1070, permute_811, view_1067, permute_806, view_1063, permute_794, view_1053, sum_217, sum_218, permute_790, view_1050, permute_786, view_1047, sum_211, sum_212, permute_782, view_1044, permute_778, view_1041, permute_773, view_1037, permute_761, view_1025, sum_202, sum_203, permute_757, view_1022, permute_753, view_1019, permute_748, view_1015, permute_736, view_1005, sum_193, sum_194, permute_732, view_1002, permute_728, view_999, sum_187, sum_188, permute_724, view_996, permute_720, view_993, permute_715, view_989, permute_703, view_977, sum_178, sum_179, permute_699, view_974, permute_695, view_971, permute_690, view_967, permute_678, view_957, sum_169, sum_170, permute_674, view_954, permute_670, view_951, sum_163, sum_164, permute_666, view_948, permute_662, view_945, permute_657, view_941, permute_645, view_929, sum_154, sum_155, permute_641, view_926, permute_637, view_923, permute_632, view_919, permute_620, view_909, sum_145, sum_146, permute_616, view_906, permute_612, view_903, sum_139, sum_140, permute_608, view_900, permute_604, view_897, permute_599, view_893, permute_587, view_881, sum_130, sum_131, permute_583, view_878, permute_579, view_875, permute_574, view_871, permute_562, view_861, sum_121, sum_122, permute_558, view_858, permute_554, view_855, sum_115, sum_116, permute_550, view_852, permute_546, view_849, permute_541, view_845, permute_529, view_833, sum_106, sum_107, permute_525, view_830, permute_521, view_827, permute_516, view_823, permute_504, view_813, sum_97, sum_98, permute_500, view_810, permute_496, view_807, sum_91, sum_92, permute_492, view_804, permute_488, view_801, permute_483, view_797, permute_471, view_785, sum_82, sum_83, permute_467, view_782, permute_463, view_779, permute_458, view_775, permute_446, view_765, sum_73, sum_74, permute_442, view_762, permute_438, view_759, sum_67, sum_68, permute_434, view_756, permute_430, view_753, permute_425, view_749, permute_413, view_737, sum_58, sum_59, permute_409, view_734, permute_405, view_731, permute_400, view_727, permute_388, view_717, sum_49, sum_50, permute_384, view_714, permute_380, view_711, sum_43, sum_44, permute_376, None, None, None, None, None]
    