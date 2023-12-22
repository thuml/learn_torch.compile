from __future__ import annotations



def forward(self, primals_3: "f32[2048]", primals_10: "f32[2048]", primals_16: "f32[2048]", primals_23: "f32[2048]", primals_29: "f32[2048]", primals_36: "f32[2048]", primals_42: "f32[2048]", primals_49: "f32[2048]", primals_55: "f32[2048]", primals_62: "f32[2048]", primals_68: "f32[2048]", primals_75: "f32[2048]", primals_81: "f32[2048]", primals_88: "f32[2048]", primals_94: "f32[2048]", primals_101: "f32[2048]", primals_107: "f32[2048]", primals_114: "f32[2048]", primals_120: "f32[2048]", primals_127: "f32[2048]", primals_133: "f32[2048]", primals_140: "f32[2048]", primals_146: "f32[2048]", primals_153: "f32[2048]", primals_159: "f32[2048]", primals_166: "f32[2048]", primals_172: "f32[2048]", primals_179: "f32[2048]", primals_185: "f32[2048]", primals_192: "f32[2048]", primals_198: "f32[2048]", primals_205: "f32[2048]", primals_211: "f32[2048]", primals_218: "f32[2048]", primals_224: "f32[2048]", primals_231: "f32[2048]", primals_237: "f32[2048]", primals_244: "f32[2048]", primals_250: "f32[2048]", primals_257: "f32[2048]", primals_263: "f32[2048]", primals_270: "f32[2048]", primals_276: "f32[2048]", primals_283: "f32[2048]", primals_289: "f32[2048]", primals_296: "f32[2048]", primals_302: "f32[2048]", primals_309: "f32[2048]", primals_315: "f32[2048]", primals_343: "i64[1, 128]", view: "i64[1, 128]", view_1: "i64[1, 128]", mul: "f32[1, 128, 2048]", view_2: "f32[128, 2048]", slice_4: "b8[1, 1, 128, 128]", view_18: "f32[128, 2048]", mul_2: "f32[1, 128, 2048]", view_20: "f32[128, 2048]", addmm_1: "f32[128, 8192]", tanh: "f32[1, 128, 8192]", view_22: "f32[128, 8192]", mul_8: "f32[1, 128, 2048]", view_24: "f32[128, 2048]", slice_8: "b8[1, 1, 128, 128]", view_40: "f32[128, 2048]", mul_10: "f32[1, 128, 2048]", view_42: "f32[128, 2048]", addmm_4: "f32[128, 8192]", tanh_1: "f32[1, 128, 8192]", view_44: "f32[128, 8192]", mul_16: "f32[1, 128, 2048]", view_46: "f32[128, 2048]", slice_12: "b8[1, 1, 128, 128]", view_62: "f32[128, 2048]", mul_18: "f32[1, 128, 2048]", view_64: "f32[128, 2048]", addmm_7: "f32[128, 8192]", tanh_2: "f32[1, 128, 8192]", view_66: "f32[128, 8192]", mul_24: "f32[1, 128, 2048]", view_68: "f32[128, 2048]", slice_16: "b8[1, 1, 128, 128]", view_84: "f32[128, 2048]", mul_26: "f32[1, 128, 2048]", view_86: "f32[128, 2048]", addmm_10: "f32[128, 8192]", tanh_3: "f32[1, 128, 8192]", view_88: "f32[128, 8192]", mul_32: "f32[1, 128, 2048]", view_90: "f32[128, 2048]", slice_20: "b8[1, 1, 128, 128]", view_106: "f32[128, 2048]", mul_34: "f32[1, 128, 2048]", view_108: "f32[128, 2048]", addmm_13: "f32[128, 8192]", tanh_4: "f32[1, 128, 8192]", view_110: "f32[128, 8192]", mul_40: "f32[1, 128, 2048]", view_112: "f32[128, 2048]", slice_24: "b8[1, 1, 128, 128]", view_128: "f32[128, 2048]", mul_42: "f32[1, 128, 2048]", view_130: "f32[128, 2048]", addmm_16: "f32[128, 8192]", tanh_5: "f32[1, 128, 8192]", view_132: "f32[128, 8192]", mul_48: "f32[1, 128, 2048]", view_134: "f32[128, 2048]", slice_28: "b8[1, 1, 128, 128]", view_150: "f32[128, 2048]", mul_50: "f32[1, 128, 2048]", view_152: "f32[128, 2048]", addmm_19: "f32[128, 8192]", tanh_6: "f32[1, 128, 8192]", view_154: "f32[128, 8192]", mul_56: "f32[1, 128, 2048]", view_156: "f32[128, 2048]", slice_32: "b8[1, 1, 128, 128]", view_172: "f32[128, 2048]", mul_58: "f32[1, 128, 2048]", view_174: "f32[128, 2048]", addmm_22: "f32[128, 8192]", tanh_7: "f32[1, 128, 8192]", view_176: "f32[128, 8192]", mul_64: "f32[1, 128, 2048]", view_178: "f32[128, 2048]", slice_36: "b8[1, 1, 128, 128]", view_194: "f32[128, 2048]", mul_66: "f32[1, 128, 2048]", view_196: "f32[128, 2048]", addmm_25: "f32[128, 8192]", tanh_8: "f32[1, 128, 8192]", view_198: "f32[128, 8192]", mul_72: "f32[1, 128, 2048]", view_200: "f32[128, 2048]", slice_40: "b8[1, 1, 128, 128]", view_216: "f32[128, 2048]", mul_74: "f32[1, 128, 2048]", view_218: "f32[128, 2048]", addmm_28: "f32[128, 8192]", tanh_9: "f32[1, 128, 8192]", view_220: "f32[128, 8192]", mul_80: "f32[1, 128, 2048]", view_222: "f32[128, 2048]", slice_44: "b8[1, 1, 128, 128]", view_238: "f32[128, 2048]", mul_82: "f32[1, 128, 2048]", view_240: "f32[128, 2048]", addmm_31: "f32[128, 8192]", tanh_10: "f32[1, 128, 8192]", view_242: "f32[128, 8192]", mul_88: "f32[1, 128, 2048]", view_244: "f32[128, 2048]", slice_48: "b8[1, 1, 128, 128]", view_260: "f32[128, 2048]", mul_90: "f32[1, 128, 2048]", view_262: "f32[128, 2048]", addmm_34: "f32[128, 8192]", tanh_11: "f32[1, 128, 8192]", view_264: "f32[128, 8192]", mul_96: "f32[1, 128, 2048]", view_266: "f32[128, 2048]", slice_52: "b8[1, 1, 128, 128]", view_282: "f32[128, 2048]", mul_98: "f32[1, 128, 2048]", view_284: "f32[128, 2048]", addmm_37: "f32[128, 8192]", tanh_12: "f32[1, 128, 8192]", view_286: "f32[128, 8192]", mul_104: "f32[1, 128, 2048]", view_288: "f32[128, 2048]", slice_56: "b8[1, 1, 128, 128]", view_304: "f32[128, 2048]", mul_106: "f32[1, 128, 2048]", view_306: "f32[128, 2048]", addmm_40: "f32[128, 8192]", tanh_13: "f32[1, 128, 8192]", view_308: "f32[128, 8192]", mul_112: "f32[1, 128, 2048]", view_310: "f32[128, 2048]", slice_60: "b8[1, 1, 128, 128]", view_326: "f32[128, 2048]", mul_114: "f32[1, 128, 2048]", view_328: "f32[128, 2048]", addmm_43: "f32[128, 8192]", tanh_14: "f32[1, 128, 8192]", view_330: "f32[128, 8192]", mul_120: "f32[1, 128, 2048]", view_332: "f32[128, 2048]", slice_64: "b8[1, 1, 128, 128]", view_348: "f32[128, 2048]", mul_122: "f32[1, 128, 2048]", view_350: "f32[128, 2048]", addmm_46: "f32[128, 8192]", tanh_15: "f32[1, 128, 8192]", view_352: "f32[128, 8192]", mul_128: "f32[1, 128, 2048]", view_354: "f32[128, 2048]", slice_68: "b8[1, 1, 128, 128]", view_370: "f32[128, 2048]", mul_130: "f32[1, 128, 2048]", view_372: "f32[128, 2048]", addmm_49: "f32[128, 8192]", tanh_16: "f32[1, 128, 8192]", view_374: "f32[128, 8192]", mul_136: "f32[1, 128, 2048]", view_376: "f32[128, 2048]", slice_72: "b8[1, 1, 128, 128]", view_392: "f32[128, 2048]", mul_138: "f32[1, 128, 2048]", view_394: "f32[128, 2048]", addmm_52: "f32[128, 8192]", tanh_17: "f32[1, 128, 8192]", view_396: "f32[128, 8192]", mul_144: "f32[1, 128, 2048]", view_398: "f32[128, 2048]", slice_76: "b8[1, 1, 128, 128]", view_414: "f32[128, 2048]", mul_146: "f32[1, 128, 2048]", view_416: "f32[128, 2048]", addmm_55: "f32[128, 8192]", tanh_18: "f32[1, 128, 8192]", view_418: "f32[128, 8192]", mul_152: "f32[1, 128, 2048]", view_420: "f32[128, 2048]", slice_80: "b8[1, 1, 128, 128]", view_436: "f32[128, 2048]", mul_154: "f32[1, 128, 2048]", view_438: "f32[128, 2048]", addmm_58: "f32[128, 8192]", tanh_19: "f32[1, 128, 8192]", view_440: "f32[128, 8192]", mul_160: "f32[1, 128, 2048]", view_442: "f32[128, 2048]", slice_84: "b8[1, 1, 128, 128]", view_458: "f32[128, 2048]", mul_162: "f32[1, 128, 2048]", view_460: "f32[128, 2048]", addmm_61: "f32[128, 8192]", tanh_20: "f32[1, 128, 8192]", view_462: "f32[128, 8192]", mul_168: "f32[1, 128, 2048]", view_464: "f32[128, 2048]", slice_88: "b8[1, 1, 128, 128]", view_480: "f32[128, 2048]", mul_170: "f32[1, 128, 2048]", view_482: "f32[128, 2048]", addmm_64: "f32[128, 8192]", tanh_21: "f32[1, 128, 8192]", view_484: "f32[128, 8192]", mul_176: "f32[1, 128, 2048]", view_486: "f32[128, 2048]", slice_92: "b8[1, 1, 128, 128]", view_502: "f32[128, 2048]", mul_178: "f32[1, 128, 2048]", view_504: "f32[128, 2048]", addmm_67: "f32[128, 8192]", tanh_22: "f32[1, 128, 8192]", view_506: "f32[128, 8192]", mul_184: "f32[1, 128, 2048]", view_508: "f32[128, 2048]", slice_96: "b8[1, 1, 128, 128]", view_524: "f32[128, 2048]", mul_186: "f32[1, 128, 2048]", view_526: "f32[128, 2048]", addmm_70: "f32[128, 8192]", tanh_23: "f32[1, 128, 8192]", view_528: "f32[128, 8192]", mul_192: "f32[1, 128, 2048]", view_531: "f32[128, 2048]", sub_74: "f32[127, 50257]", convert_element_type: "f32[]", permute_267: "f32[50257, 2048]", div_26: "f32[1, 128, 1]", permute_269: "f32[2048, 8192]", permute_273: "f32[8192, 2048]", div_27: "f32[1, 128, 1]", permute_277: "f32[2048, 2048]", permute_282: "f32[16, 128, 128]", permute_283: "f32[16, 128, 128]", alias_51: "f32[1, 16, 128, 128]", permute_284: "f32[16, 128, 128]", permute_285: "f32[16, 128, 128]", permute_292: "f32[2048, 2048]", permute_296: "f32[2048, 2048]", permute_300: "f32[2048, 2048]", div_28: "f32[1, 128, 1]", permute_302: "f32[2048, 8192]", permute_306: "f32[8192, 2048]", div_29: "f32[1, 128, 1]", permute_310: "f32[2048, 2048]", permute_315: "f32[16, 128, 128]", permute_316: "f32[16, 128, 128]", alias_53: "f32[1, 16, 128, 128]", permute_317: "f32[16, 128, 128]", permute_318: "f32[16, 128, 128]", permute_325: "f32[2048, 2048]", permute_329: "f32[2048, 2048]", permute_333: "f32[2048, 2048]", div_30: "f32[1, 128, 1]", permute_335: "f32[2048, 8192]", permute_339: "f32[8192, 2048]", div_31: "f32[1, 128, 1]", permute_343: "f32[2048, 2048]", permute_348: "f32[16, 128, 128]", permute_349: "f32[16, 128, 128]", alias_55: "f32[1, 16, 128, 128]", permute_350: "f32[16, 128, 128]", permute_351: "f32[16, 128, 128]", permute_358: "f32[2048, 2048]", permute_362: "f32[2048, 2048]", permute_366: "f32[2048, 2048]", div_32: "f32[1, 128, 1]", permute_368: "f32[2048, 8192]", permute_372: "f32[8192, 2048]", div_33: "f32[1, 128, 1]", permute_376: "f32[2048, 2048]", permute_381: "f32[16, 128, 128]", permute_382: "f32[16, 128, 128]", alias_57: "f32[1, 16, 128, 128]", permute_383: "f32[16, 128, 128]", permute_384: "f32[16, 128, 128]", permute_391: "f32[2048, 2048]", permute_395: "f32[2048, 2048]", permute_399: "f32[2048, 2048]", div_34: "f32[1, 128, 1]", permute_401: "f32[2048, 8192]", permute_405: "f32[8192, 2048]", div_35: "f32[1, 128, 1]", permute_409: "f32[2048, 2048]", permute_414: "f32[16, 128, 128]", permute_415: "f32[16, 128, 128]", alias_59: "f32[1, 16, 128, 128]", permute_416: "f32[16, 128, 128]", permute_417: "f32[16, 128, 128]", permute_424: "f32[2048, 2048]", permute_428: "f32[2048, 2048]", permute_432: "f32[2048, 2048]", div_36: "f32[1, 128, 1]", permute_434: "f32[2048, 8192]", permute_438: "f32[8192, 2048]", div_37: "f32[1, 128, 1]", permute_442: "f32[2048, 2048]", permute_447: "f32[16, 128, 128]", permute_448: "f32[16, 128, 128]", alias_61: "f32[1, 16, 128, 128]", permute_449: "f32[16, 128, 128]", permute_450: "f32[16, 128, 128]", permute_457: "f32[2048, 2048]", permute_461: "f32[2048, 2048]", permute_465: "f32[2048, 2048]", div_38: "f32[1, 128, 1]", permute_467: "f32[2048, 8192]", permute_471: "f32[8192, 2048]", div_39: "f32[1, 128, 1]", permute_475: "f32[2048, 2048]", permute_480: "f32[16, 128, 128]", permute_481: "f32[16, 128, 128]", alias_63: "f32[1, 16, 128, 128]", permute_482: "f32[16, 128, 128]", permute_483: "f32[16, 128, 128]", permute_490: "f32[2048, 2048]", permute_494: "f32[2048, 2048]", permute_498: "f32[2048, 2048]", div_40: "f32[1, 128, 1]", permute_500: "f32[2048, 8192]", permute_504: "f32[8192, 2048]", div_41: "f32[1, 128, 1]", permute_508: "f32[2048, 2048]", permute_513: "f32[16, 128, 128]", permute_514: "f32[16, 128, 128]", alias_65: "f32[1, 16, 128, 128]", permute_515: "f32[16, 128, 128]", permute_516: "f32[16, 128, 128]", permute_523: "f32[2048, 2048]", permute_527: "f32[2048, 2048]", permute_531: "f32[2048, 2048]", div_42: "f32[1, 128, 1]", permute_533: "f32[2048, 8192]", permute_537: "f32[8192, 2048]", div_43: "f32[1, 128, 1]", permute_541: "f32[2048, 2048]", permute_546: "f32[16, 128, 128]", permute_547: "f32[16, 128, 128]", alias_67: "f32[1, 16, 128, 128]", permute_548: "f32[16, 128, 128]", permute_549: "f32[16, 128, 128]", permute_556: "f32[2048, 2048]", permute_560: "f32[2048, 2048]", permute_564: "f32[2048, 2048]", div_44: "f32[1, 128, 1]", permute_566: "f32[2048, 8192]", permute_570: "f32[8192, 2048]", div_45: "f32[1, 128, 1]", permute_574: "f32[2048, 2048]", permute_579: "f32[16, 128, 128]", permute_580: "f32[16, 128, 128]", alias_69: "f32[1, 16, 128, 128]", permute_581: "f32[16, 128, 128]", permute_582: "f32[16, 128, 128]", permute_589: "f32[2048, 2048]", permute_593: "f32[2048, 2048]", permute_597: "f32[2048, 2048]", div_46: "f32[1, 128, 1]", permute_599: "f32[2048, 8192]", permute_603: "f32[8192, 2048]", div_47: "f32[1, 128, 1]", permute_607: "f32[2048, 2048]", permute_612: "f32[16, 128, 128]", permute_613: "f32[16, 128, 128]", alias_71: "f32[1, 16, 128, 128]", permute_614: "f32[16, 128, 128]", permute_615: "f32[16, 128, 128]", permute_622: "f32[2048, 2048]", permute_626: "f32[2048, 2048]", permute_630: "f32[2048, 2048]", div_48: "f32[1, 128, 1]", permute_632: "f32[2048, 8192]", permute_636: "f32[8192, 2048]", div_49: "f32[1, 128, 1]", permute_640: "f32[2048, 2048]", permute_645: "f32[16, 128, 128]", permute_646: "f32[16, 128, 128]", alias_73: "f32[1, 16, 128, 128]", permute_647: "f32[16, 128, 128]", permute_648: "f32[16, 128, 128]", permute_655: "f32[2048, 2048]", permute_659: "f32[2048, 2048]", permute_663: "f32[2048, 2048]", div_50: "f32[1, 128, 1]", permute_665: "f32[2048, 8192]", permute_669: "f32[8192, 2048]", div_51: "f32[1, 128, 1]", permute_673: "f32[2048, 2048]", permute_678: "f32[16, 128, 128]", permute_679: "f32[16, 128, 128]", alias_75: "f32[1, 16, 128, 128]", permute_680: "f32[16, 128, 128]", permute_681: "f32[16, 128, 128]", permute_688: "f32[2048, 2048]", permute_692: "f32[2048, 2048]", permute_696: "f32[2048, 2048]", div_52: "f32[1, 128, 1]", permute_698: "f32[2048, 8192]", permute_702: "f32[8192, 2048]", div_53: "f32[1, 128, 1]", permute_706: "f32[2048, 2048]", permute_711: "f32[16, 128, 128]", permute_712: "f32[16, 128, 128]", alias_77: "f32[1, 16, 128, 128]", permute_713: "f32[16, 128, 128]", permute_714: "f32[16, 128, 128]", permute_721: "f32[2048, 2048]", permute_725: "f32[2048, 2048]", permute_729: "f32[2048, 2048]", div_54: "f32[1, 128, 1]", permute_731: "f32[2048, 8192]", permute_735: "f32[8192, 2048]", div_55: "f32[1, 128, 1]", permute_739: "f32[2048, 2048]", permute_744: "f32[16, 128, 128]", permute_745: "f32[16, 128, 128]", alias_79: "f32[1, 16, 128, 128]", permute_746: "f32[16, 128, 128]", permute_747: "f32[16, 128, 128]", permute_754: "f32[2048, 2048]", permute_758: "f32[2048, 2048]", permute_762: "f32[2048, 2048]", div_56: "f32[1, 128, 1]", permute_764: "f32[2048, 8192]", permute_768: "f32[8192, 2048]", div_57: "f32[1, 128, 1]", permute_772: "f32[2048, 2048]", permute_777: "f32[16, 128, 128]", permute_778: "f32[16, 128, 128]", alias_81: "f32[1, 16, 128, 128]", permute_779: "f32[16, 128, 128]", permute_780: "f32[16, 128, 128]", permute_787: "f32[2048, 2048]", permute_791: "f32[2048, 2048]", permute_795: "f32[2048, 2048]", div_58: "f32[1, 128, 1]", permute_797: "f32[2048, 8192]", permute_801: "f32[8192, 2048]", div_59: "f32[1, 128, 1]", permute_805: "f32[2048, 2048]", permute_810: "f32[16, 128, 128]", permute_811: "f32[16, 128, 128]", alias_83: "f32[1, 16, 128, 128]", permute_812: "f32[16, 128, 128]", permute_813: "f32[16, 128, 128]", permute_820: "f32[2048, 2048]", permute_824: "f32[2048, 2048]", permute_828: "f32[2048, 2048]", div_60: "f32[1, 128, 1]", permute_830: "f32[2048, 8192]", permute_834: "f32[8192, 2048]", div_61: "f32[1, 128, 1]", permute_838: "f32[2048, 2048]", permute_843: "f32[16, 128, 128]", permute_844: "f32[16, 128, 128]", alias_85: "f32[1, 16, 128, 128]", permute_845: "f32[16, 128, 128]", permute_846: "f32[16, 128, 128]", permute_853: "f32[2048, 2048]", permute_857: "f32[2048, 2048]", permute_861: "f32[2048, 2048]", div_62: "f32[1, 128, 1]", permute_863: "f32[2048, 8192]", permute_867: "f32[8192, 2048]", div_63: "f32[1, 128, 1]", permute_871: "f32[2048, 2048]", permute_876: "f32[16, 128, 128]", permute_877: "f32[16, 128, 128]", alias_87: "f32[1, 16, 128, 128]", permute_878: "f32[16, 128, 128]", permute_879: "f32[16, 128, 128]", permute_886: "f32[2048, 2048]", permute_890: "f32[2048, 2048]", permute_894: "f32[2048, 2048]", div_64: "f32[1, 128, 1]", permute_896: "f32[2048, 8192]", permute_900: "f32[8192, 2048]", div_65: "f32[1, 128, 1]", permute_904: "f32[2048, 2048]", permute_909: "f32[16, 128, 128]", permute_910: "f32[16, 128, 128]", alias_89: "f32[1, 16, 128, 128]", permute_911: "f32[16, 128, 128]", permute_912: "f32[16, 128, 128]", permute_919: "f32[2048, 2048]", permute_923: "f32[2048, 2048]", permute_927: "f32[2048, 2048]", div_66: "f32[1, 128, 1]", permute_929: "f32[2048, 8192]", permute_933: "f32[8192, 2048]", div_67: "f32[1, 128, 1]", permute_937: "f32[2048, 2048]", permute_942: "f32[16, 128, 128]", permute_943: "f32[16, 128, 128]", alias_91: "f32[1, 16, 128, 128]", permute_944: "f32[16, 128, 128]", permute_945: "f32[16, 128, 128]", permute_952: "f32[2048, 2048]", permute_956: "f32[2048, 2048]", permute_960: "f32[2048, 2048]", div_68: "f32[1, 128, 1]", permute_962: "f32[2048, 8192]", permute_966: "f32[8192, 2048]", div_69: "f32[1, 128, 1]", permute_970: "f32[2048, 2048]", permute_975: "f32[16, 128, 128]", permute_976: "f32[16, 128, 128]", alias_93: "f32[1, 16, 128, 128]", permute_977: "f32[16, 128, 128]", permute_978: "f32[16, 128, 128]", permute_985: "f32[2048, 2048]", permute_989: "f32[2048, 2048]", permute_993: "f32[2048, 2048]", div_70: "f32[1, 128, 1]", permute_995: "f32[2048, 8192]", permute_999: "f32[8192, 2048]", div_71: "f32[1, 128, 1]", permute_1003: "f32[2048, 2048]", permute_1008: "f32[16, 128, 128]", permute_1009: "f32[16, 128, 128]", alias_95: "f32[1, 16, 128, 128]", permute_1010: "f32[16, 128, 128]", permute_1011: "f32[16, 128, 128]", permute_1018: "f32[2048, 2048]", permute_1022: "f32[2048, 2048]", permute_1026: "f32[2048, 2048]", div_72: "f32[1, 128, 1]", permute_1028: "f32[2048, 8192]", permute_1032: "f32[8192, 2048]", div_73: "f32[1, 128, 1]", permute_1036: "f32[2048, 2048]", permute_1041: "f32[16, 128, 128]", permute_1042: "f32[16, 128, 128]", alias_97: "f32[1, 16, 128, 128]", permute_1043: "f32[16, 128, 128]", permute_1044: "f32[16, 128, 128]", permute_1051: "f32[2048, 2048]", permute_1055: "f32[2048, 2048]", permute_1059: "f32[2048, 2048]", div_74: "f32[1, 128, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 50257]", tangents_3: "f32[1, 16, 128, 128]", tangents_4: "f32[1, 16, 128, 128]", tangents_5: "f32[1, 16, 128, 128]", tangents_6: "f32[1, 16, 128, 128]", tangents_7: "f32[1, 16, 128, 128]", tangents_8: "f32[1, 16, 128, 128]", tangents_9: "f32[1, 16, 128, 128]", tangents_10: "f32[1, 16, 128, 128]", tangents_11: "f32[1, 16, 128, 128]", tangents_12: "f32[1, 16, 128, 128]", tangents_13: "f32[1, 16, 128, 128]", tangents_14: "f32[1, 16, 128, 128]", tangents_15: "f32[1, 16, 128, 128]", tangents_16: "f32[1, 16, 128, 128]", tangents_17: "f32[1, 16, 128, 128]", tangents_18: "f32[1, 16, 128, 128]", tangents_19: "f32[1, 16, 128, 128]", tangents_20: "f32[1, 16, 128, 128]", tangents_21: "f32[1, 16, 128, 128]", tangents_22: "f32[1, 16, 128, 128]", tangents_23: "f32[1, 16, 128, 128]", tangents_24: "f32[1, 16, 128, 128]", tangents_25: "f32[1, 16, 128, 128]", tangents_26: "f32[1, 16, 128, 128]", tangents_27: "f32[1, 16, 128, 128]", tangents_28: "f32[1, 16, 128, 128]", tangents_29: "f32[1, 16, 128, 128]", tangents_30: "f32[1, 16, 128, 128]", tangents_31: "f32[1, 16, 128, 128]", tangents_32: "f32[1, 16, 128, 128]", tangents_33: "f32[1, 16, 128, 128]", tangents_34: "f32[1, 16, 128, 128]", tangents_35: "f32[1, 16, 128, 128]", tangents_36: "f32[1, 16, 128, 128]", tangents_37: "f32[1, 16, 128, 128]", tangents_38: "f32[1, 16, 128, 128]", tangents_39: "f32[1, 16, 128, 128]", tangents_40: "f32[1, 16, 128, 128]", tangents_41: "f32[1, 16, 128, 128]", tangents_42: "f32[1, 16, 128, 128]", tangents_43: "f32[1, 16, 128, 128]", tangents_44: "f32[1, 16, 128, 128]", tangents_45: "f32[1, 16, 128, 128]", tangents_46: "f32[1, 16, 128, 128]", tangents_47: "f32[1, 16, 128, 128]", tangents_48: "f32[1, 16, 128, 128]", tangents_49: "f32[1, 16, 128, 128]", tangents_50: "f32[1, 16, 128, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_21: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_1, [1, 128, 8192]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    alias_1: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh)
    add_7: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_43: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_4, [1, 128, 8192]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    alias_3: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_1)
    add_15: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_65: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_7, [1, 128, 8192]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    alias_5: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_2)
    add_23: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_87: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_10, [1, 128, 8192]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    alias_7: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_3)
    add_31: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_109: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_13, [1, 128, 8192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    alias_9: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_4)
    add_39: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_131: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_16, [1, 128, 8192]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    alias_11: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_5)
    add_47: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_153: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_19, [1, 128, 8192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    alias_13: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_6)
    add_55: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_175: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_22, [1, 128, 8192]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    alias_15: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_7)
    add_63: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_197: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_25, [1, 128, 8192]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    alias_17: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_8)
    add_71: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_219: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_28, [1, 128, 8192]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    alias_19: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_9)
    add_79: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_241: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_31, [1, 128, 8192]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    alias_21: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_10)
    add_87: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_263: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_34, [1, 128, 8192]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    alias_23: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_11)
    add_95: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_285: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_37, [1, 128, 8192]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    alias_25: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_12)
    add_103: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_307: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_40, [1, 128, 8192]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_108: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    alias_27: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_13)
    add_111: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_329: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_43, [1, 128, 8192]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_329, 0.5)
    alias_29: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_14)
    add_119: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_351: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_46, [1, 128, 8192]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_124: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_351, 0.5)
    alias_31: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_15)
    add_127: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_373: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_49, [1, 128, 8192]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_132: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_373, 0.5)
    alias_33: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_16)
    add_135: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_395: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_52, [1, 128, 8192]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_140: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_395, 0.5)
    alias_35: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_17)
    add_143: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_417: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_55, [1, 128, 8192]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_148: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    alias_37: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_18)
    add_151: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_439: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_58, [1, 128, 8192]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_439, 0.5)
    alias_39: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_19)
    add_159: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_461: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_61, [1, 128, 8192]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_461, 0.5)
    alias_41: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_20)
    add_167: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_483: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_64, [1, 128, 8192]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_172: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
    alias_43: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_21)
    add_175: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_505: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_67, [1, 128, 8192]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_180: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_505, 0.5)
    alias_45: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_22)
    add_183: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_527: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_70, [1, 128, 8192]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_188: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    alias_47: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_23)
    add_191: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:775, code: shift_labels = labels[..., 1:].contiguous()
    slice_99: "i64[1, 127]" = torch.ops.aten.slice.Tensor(primals_343, 1, 1, 9223372036854775807);  primals_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:778, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_534: "i64[127]" = torch.ops.aten.view.default(slice_99, [-1]);  slice_99 = None
    alias_48: "f32[127, 50257]" = torch.ops.aten.alias.default(sub_74);  sub_74 = None
    full_default_24: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_2: "i64[127, 1]" = torch.ops.aten.unsqueeze.default(view_534, 1);  view_534 = None
    ne_3: "b8[127, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_2, -100)
    where_26: "i64[127, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_2, full_default_24);  unsqueeze_2 = full_default_24 = None
    full_default_27: "f32[127, 50257]" = torch.ops.aten.full.default([127, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[127, 50257]" = torch.ops.aten.scatter.value(full_default_27, 1, where_26, -1.0);  full_default_27 = where_26 = None
    where_27: "f32[127, 1]" = torch.ops.aten.where.self(ne_3, div_25, full_default_25);  ne_3 = div_25 = None
    mul_194: "f32[127, 50257]" = torch.ops.aten.mul.Tensor(scatter, where_27);  scatter = where_27 = None
    alias_49: "f32[127, 50257]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    exp_25: "f32[127, 50257]" = torch.ops.aten.exp.default(alias_49);  alias_49 = None
    sum_28: "f32[127, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [1], True)
    mul_195: "f32[127, 50257]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_75: "f32[127, 50257]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    view_535: "f32[1, 127, 50257]" = torch.ops.aten.view.default(sub_75, [1, 127, 50257]);  sub_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:774, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    full_default_29: "f32[1, 127, 50257]" = torch.ops.aten.full.default([1, 127, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[1, 127, 50257]" = torch.ops.aten.slice_scatter.default(full_default_29, view_535, 2, 0, 9223372036854775807);  full_default_29 = view_535 = None
    full_default_30: "f32[1, 128, 50257]" = torch.ops.aten.full.default([1, 128, 50257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 128, 50257]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter, 1, 0, -1);  full_default_30 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:774, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    add_195: "f32[1, 128, 50257]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:763, code: lm_logits = self.lm_head(hidden_states)
    view_536: "f32[128, 50257]" = torch.ops.aten.view.default(add_195, [128, 50257]);  add_195 = None
    permute_265: "f32[50257, 128]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_73: "f32[50257, 2048]" = torch.ops.aten.mm.default(permute_265, view_531);  permute_265 = view_531 = None
    permute_266: "f32[2048, 50257]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    mm_74: "f32[128, 2048]" = torch.ops.aten.mm.default(view_536, permute_267);  view_536 = permute_267 = None
    view_537: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_74, [1, 128, 2048]);  mm_74 = None
    permute_268: "f32[50257, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    mul_197: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_537, primals_315);  primals_315 = None
    mul_198: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_197, 2048)
    sum_29: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
    mul_199: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_197, mul_192);  mul_197 = None
    sum_30: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    mul_200: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, sum_30);  sum_30 = None
    sub_77: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_198, sum_29);  mul_198 = sum_29 = None
    sub_78: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_77, mul_200);  sub_77 = mul_200 = None
    mul_201: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_26, sub_78);  div_26 = sub_78 = None
    mul_202: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_537, mul_192);  mul_192 = None
    sum_31: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
    sum_32: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_539: "f32[128, 2048]" = torch.ops.aten.view.default(mul_201, [128, 2048])
    mm_75: "f32[128, 8192]" = torch.ops.aten.mm.default(view_539, permute_269);  permute_269 = None
    permute_270: "f32[2048, 128]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_76: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_270, view_528);  permute_270 = view_528 = None
    permute_271: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_33: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[2048]" = torch.ops.aten.view.default(sum_33, [2048]);  sum_33 = None
    permute_272: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_541: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_75, [1, 128, 8192]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_203: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_541, mul_188);  mul_188 = None
    mul_204: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_541, add_191);  view_541 = add_191 = None
    alias_50: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    mul_205: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_50, alias_50);  alias_50 = None
    sub_79: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_205);  mul_205 = None
    mul_206: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_203, sub_79);  mul_203 = sub_79 = None
    mul_207: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_206, 0.7978845608028654);  mul_206 = None
    mul_208: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_207, 0.044715)
    pow_25: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 2.0);  view_527 = None
    mul_209: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_210: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_196: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_207, mul_210);  mul_207 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_211: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_204, 0.5);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_197: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_196, mul_211);  add_196 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_542: "f32[128, 8192]" = torch.ops.aten.view.default(add_197, [128, 8192]);  add_197 = None
    mm_77: "f32[128, 2048]" = torch.ops.aten.mm.default(view_542, permute_273);  permute_273 = None
    permute_274: "f32[8192, 128]" = torch.ops.aten.permute.default(view_542, [1, 0])
    mm_78: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_274, view_526);  permute_274 = view_526 = None
    permute_275: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_34: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
    view_543: "f32[8192]" = torch.ops.aten.view.default(sum_34, [8192]);  sum_34 = None
    permute_276: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_544: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_77, [1, 128, 2048]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_213: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_544, primals_309);  primals_309 = None
    mul_214: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_213, 2048)
    sum_35: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_213, mul_186);  mul_213 = None
    sum_36: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, sum_36);  sum_36 = None
    sub_81: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_214, sum_35);  mul_214 = sum_35 = None
    sub_82: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_81, mul_216);  sub_81 = mul_216 = None
    mul_217: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
    mul_218: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_544, mul_186);  mul_186 = None
    sum_37: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_38: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_544, [0, 1]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_198: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_201, mul_217);  mul_201 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_545: "f32[128, 2048]" = torch.ops.aten.view.default(add_198, [128, 2048])
    mm_79: "f32[128, 2048]" = torch.ops.aten.mm.default(view_545, permute_277);  permute_277 = None
    permute_278: "f32[2048, 128]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_80: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_278, view_524);  permute_278 = view_524 = None
    permute_279: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_39: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[2048]" = torch.ops.aten.view.default(sum_39, [2048]);  sum_39 = None
    permute_280: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_547: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_79, [1, 128, 2048]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_548: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_547, [1, 128, 16, 128]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_281: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_549: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_281, [16, 128, 128]);  permute_281 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_282, view_549);  permute_282 = None
    bmm_49: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_549, permute_283);  view_549 = permute_283 = None
    view_550: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_199: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_50, view_550);  tangents_50 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_551: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, [1, 16, 128, 128]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_219: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_551, alias_51);  view_551 = None
    sum_40: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_51, sum_40);  alias_51 = sum_40 = None
    sub_83: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_28: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, sub_83, full_default_25);  slice_96 = sub_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_552: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_28, [16, 128, 128]);  where_28 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_284, view_552);  permute_284 = None
    bmm_51: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_552, permute_285);  view_552 = permute_285 = None
    view_553: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, [1, 16, 128, 128]);  bmm_50 = None
    view_554: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_51, [1, 16, 128, 128]);  bmm_51 = None
    permute_286: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_553, [0, 1, 3, 2]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_200: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_49, permute_286);  tangents_49 = permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_287: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_199, [0, 2, 1, 3]);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_97: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_555: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_97, [1, 128, 2048]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_288: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_200, [0, 2, 1, 3]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_98: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_556: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_98, [1, 128, 2048]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_289: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_99: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_557: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_99, [1, 128, 2048]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_558: "f32[128, 2048]" = torch.ops.aten.view.default(view_555, [128, 2048]);  view_555 = None
    permute_290: "f32[2048, 128]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_81: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_290, view_508);  permute_290 = None
    permute_291: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    mm_82: "f32[128, 2048]" = torch.ops.aten.mm.default(view_558, permute_292);  view_558 = permute_292 = None
    view_559: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_82, [1, 128, 2048]);  mm_82 = None
    permute_293: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_560: "f32[128, 2048]" = torch.ops.aten.view.default(view_556, [128, 2048]);  view_556 = None
    permute_294: "f32[2048, 128]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_83: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_294, view_508);  permute_294 = None
    permute_295: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    mm_84: "f32[128, 2048]" = torch.ops.aten.mm.default(view_560, permute_296);  view_560 = permute_296 = None
    view_561: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_84, [1, 128, 2048]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_201: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_559, view_561);  view_559 = view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_297: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_562: "f32[128, 2048]" = torch.ops.aten.view.default(view_557, [128, 2048]);  view_557 = None
    permute_298: "f32[2048, 128]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_85: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_298, view_508);  permute_298 = view_508 = None
    permute_299: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    mm_86: "f32[128, 2048]" = torch.ops.aten.mm.default(view_562, permute_300);  view_562 = permute_300 = None
    view_563: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_86, [1, 128, 2048]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_202: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_201, view_563);  add_201 = view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_301: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_222: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_202, primals_302);  primals_302 = None
    mul_223: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_222, 2048)
    sum_41: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_222, mul_184);  mul_222 = None
    sum_42: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, sum_42);  sum_42 = None
    sub_85: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_223, sum_41);  mul_223 = sum_41 = None
    sub_86: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_85, mul_225);  sub_85 = mul_225 = None
    mul_226: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_28, sub_86);  div_28 = sub_86 = None
    mul_227: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_202, mul_184);  mul_184 = None
    sum_43: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_44: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1]);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_203: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_198, mul_226);  add_198 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_564: "f32[128, 2048]" = torch.ops.aten.view.default(add_203, [128, 2048])
    mm_87: "f32[128, 8192]" = torch.ops.aten.mm.default(view_564, permute_302);  permute_302 = None
    permute_303: "f32[2048, 128]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_88: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_303, view_506);  permute_303 = view_506 = None
    permute_304: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_45: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[2048]" = torch.ops.aten.view.default(sum_45, [2048]);  sum_45 = None
    permute_305: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_566: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_87, [1, 128, 8192]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_228: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_566, mul_180);  mul_180 = None
    mul_229: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_566, add_183);  view_566 = add_183 = None
    alias_52: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    mul_230: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_52, alias_52);  alias_52 = None
    sub_87: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_230);  mul_230 = None
    mul_231: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_228, sub_87);  mul_228 = sub_87 = None
    mul_232: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_231, 0.7978845608028654);  mul_231 = None
    mul_233: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_232, 0.044715)
    pow_26: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_505, 2.0);  view_505 = None
    mul_234: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_235: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_204: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_232, mul_235);  mul_232 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_229, 0.5);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_205: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_204, mul_236);  add_204 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_567: "f32[128, 8192]" = torch.ops.aten.view.default(add_205, [128, 8192]);  add_205 = None
    mm_89: "f32[128, 2048]" = torch.ops.aten.mm.default(view_567, permute_306);  permute_306 = None
    permute_307: "f32[8192, 128]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_90: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_307, view_504);  permute_307 = view_504 = None
    permute_308: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_46: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[8192]" = torch.ops.aten.view.default(sum_46, [8192]);  sum_46 = None
    permute_309: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_569: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_89, [1, 128, 2048]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_238: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_569, primals_296);  primals_296 = None
    mul_239: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_238, 2048)
    sum_47: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_238, mul_178);  mul_238 = None
    sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, sum_48);  sum_48 = None
    sub_89: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_239, sum_47);  mul_239 = sum_47 = None
    sub_90: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_89, mul_241);  sub_89 = mul_241 = None
    mul_242: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_29, sub_90);  div_29 = sub_90 = None
    mul_243: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_569, mul_178);  mul_178 = None
    sum_49: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_50: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_569, [0, 1]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_206: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_203, mul_242);  add_203 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_570: "f32[128, 2048]" = torch.ops.aten.view.default(add_206, [128, 2048])
    mm_91: "f32[128, 2048]" = torch.ops.aten.mm.default(view_570, permute_310);  permute_310 = None
    permute_311: "f32[2048, 128]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_92: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_311, view_502);  permute_311 = view_502 = None
    permute_312: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_51: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[2048]" = torch.ops.aten.view.default(sum_51, [2048]);  sum_51 = None
    permute_313: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_572: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_91, [1, 128, 2048]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_573: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_572, [1, 128, 16, 128]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_314: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_574: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_314, [16, 128, 128]);  permute_314 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_315, view_574);  permute_315 = None
    bmm_53: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_574, permute_316);  view_574 = permute_316 = None
    view_575: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_207: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_48, view_575);  tangents_48 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_576: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, [1, 16, 128, 128]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_244: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_576, alias_53);  view_576 = None
    sum_52: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [-1], True)
    mul_245: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_53, sum_52);  alias_53 = sum_52 = None
    sub_91: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_29: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_92, sub_91, full_default_25);  slice_92 = sub_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_577: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_29, [16, 128, 128]);  where_29 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_317, view_577);  permute_317 = None
    bmm_55: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_577, permute_318);  view_577 = permute_318 = None
    view_578: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, [1, 16, 128, 128]);  bmm_54 = None
    view_579: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_55, [1, 16, 128, 128]);  bmm_55 = None
    permute_319: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_578, [0, 1, 3, 2]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_208: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_47, permute_319);  tangents_47 = permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_320: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_207, [0, 2, 1, 3]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_100: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_580: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_100, [1, 128, 2048]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_321: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_208, [0, 2, 1, 3]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_101: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_581: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_101, [1, 128, 2048]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_322: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_102: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_582: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_102, [1, 128, 2048]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_583: "f32[128, 2048]" = torch.ops.aten.view.default(view_580, [128, 2048]);  view_580 = None
    permute_323: "f32[2048, 128]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_93: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_323, view_486);  permute_323 = None
    permute_324: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    mm_94: "f32[128, 2048]" = torch.ops.aten.mm.default(view_583, permute_325);  view_583 = permute_325 = None
    view_584: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_94, [1, 128, 2048]);  mm_94 = None
    permute_326: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_585: "f32[128, 2048]" = torch.ops.aten.view.default(view_581, [128, 2048]);  view_581 = None
    permute_327: "f32[2048, 128]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_95: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_327, view_486);  permute_327 = None
    permute_328: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    mm_96: "f32[128, 2048]" = torch.ops.aten.mm.default(view_585, permute_329);  view_585 = permute_329 = None
    view_586: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_96, [1, 128, 2048]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_209: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_584, view_586);  view_584 = view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_330: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_587: "f32[128, 2048]" = torch.ops.aten.view.default(view_582, [128, 2048]);  view_582 = None
    permute_331: "f32[2048, 128]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_97: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_331, view_486);  permute_331 = view_486 = None
    permute_332: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    mm_98: "f32[128, 2048]" = torch.ops.aten.mm.default(view_587, permute_333);  view_587 = permute_333 = None
    view_588: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_98, [1, 128, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_210: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_209, view_588);  add_209 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_334: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_247: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_210, primals_289);  primals_289 = None
    mul_248: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_247, 2048)
    sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_247, mul_176);  mul_247 = None
    sum_54: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, sum_54);  sum_54 = None
    sub_93: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_248, sum_53);  mul_248 = sum_53 = None
    sub_94: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_93, mul_250);  sub_93 = mul_250 = None
    mul_251: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
    mul_252: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_210, mul_176);  mul_176 = None
    sum_55: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_56: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_210, [0, 1]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_211: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_206, mul_251);  add_206 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_589: "f32[128, 2048]" = torch.ops.aten.view.default(add_211, [128, 2048])
    mm_99: "f32[128, 8192]" = torch.ops.aten.mm.default(view_589, permute_335);  permute_335 = None
    permute_336: "f32[2048, 128]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_100: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_336, view_484);  permute_336 = view_484 = None
    permute_337: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_57: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[2048]" = torch.ops.aten.view.default(sum_57, [2048]);  sum_57 = None
    permute_338: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_591: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_99, [1, 128, 8192]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_253: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_591, mul_172);  mul_172 = None
    mul_254: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_591, add_175);  view_591 = add_175 = None
    alias_54: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    mul_255: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_54, alias_54);  alias_54 = None
    sub_95: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_255);  mul_255 = None
    mul_256: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_253, sub_95);  mul_253 = sub_95 = None
    mul_257: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_256, 0.7978845608028654);  mul_256 = None
    mul_258: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_257, 0.044715)
    pow_27: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_483, 2.0);  view_483 = None
    mul_259: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_27, 3.0);  pow_27 = None
    mul_260: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_212: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_257, mul_260);  mul_257 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_261: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_254, 0.5);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_213: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_212, mul_261);  add_212 = mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_592: "f32[128, 8192]" = torch.ops.aten.view.default(add_213, [128, 8192]);  add_213 = None
    mm_101: "f32[128, 2048]" = torch.ops.aten.mm.default(view_592, permute_339);  permute_339 = None
    permute_340: "f32[8192, 128]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_102: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_340, view_482);  permute_340 = view_482 = None
    permute_341: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_58: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[8192]" = torch.ops.aten.view.default(sum_58, [8192]);  sum_58 = None
    permute_342: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_594: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_101, [1, 128, 2048]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_263: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_594, primals_283);  primals_283 = None
    mul_264: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_263, 2048)
    sum_59: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True)
    mul_265: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_263, mul_170);  mul_263 = None
    sum_60: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    mul_266: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, sum_60);  sum_60 = None
    sub_97: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_264, sum_59);  mul_264 = sum_59 = None
    sub_98: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_97, mul_266);  sub_97 = mul_266 = None
    mul_267: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_31, sub_98);  div_31 = sub_98 = None
    mul_268: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_594, mul_170);  mul_170 = None
    sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    sum_62: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_594, [0, 1]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_214: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_211, mul_267);  add_211 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_595: "f32[128, 2048]" = torch.ops.aten.view.default(add_214, [128, 2048])
    mm_103: "f32[128, 2048]" = torch.ops.aten.mm.default(view_595, permute_343);  permute_343 = None
    permute_344: "f32[2048, 128]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_104: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_344, view_480);  permute_344 = view_480 = None
    permute_345: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_63: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[2048]" = torch.ops.aten.view.default(sum_63, [2048]);  sum_63 = None
    permute_346: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    view_597: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_103, [1, 128, 2048]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_598: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_597, [1, 128, 16, 128]);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_347: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_599: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_347, [16, 128, 128]);  permute_347 = None
    bmm_56: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_348, view_599);  permute_348 = None
    bmm_57: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_599, permute_349);  view_599 = permute_349 = None
    view_600: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_56, [1, 16, 128, 128]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_215: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_46, view_600);  tangents_46 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_601: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_269: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_601, alias_55);  view_601 = None
    sum_64: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [-1], True)
    mul_270: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_55, sum_64);  alias_55 = sum_64 = None
    sub_99: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_30: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_88, sub_99, full_default_25);  slice_88 = sub_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_602: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_30, [16, 128, 128]);  where_30 = None
    bmm_58: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_350, view_602);  permute_350 = None
    bmm_59: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_602, permute_351);  view_602 = permute_351 = None
    view_603: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_58, [1, 16, 128, 128]);  bmm_58 = None
    view_604: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_59, [1, 16, 128, 128]);  bmm_59 = None
    permute_352: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_603, [0, 1, 3, 2]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_216: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_45, permute_352);  tangents_45 = permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_353: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_215, [0, 2, 1, 3]);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_103: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_605: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_103, [1, 128, 2048]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_354: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_216, [0, 2, 1, 3]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_104: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_606: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_104, [1, 128, 2048]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_355: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_105: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_607: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_105, [1, 128, 2048]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_608: "f32[128, 2048]" = torch.ops.aten.view.default(view_605, [128, 2048]);  view_605 = None
    permute_356: "f32[2048, 128]" = torch.ops.aten.permute.default(view_608, [1, 0])
    mm_105: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_356, view_464);  permute_356 = None
    permute_357: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    mm_106: "f32[128, 2048]" = torch.ops.aten.mm.default(view_608, permute_358);  view_608 = permute_358 = None
    view_609: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_106, [1, 128, 2048]);  mm_106 = None
    permute_359: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_610: "f32[128, 2048]" = torch.ops.aten.view.default(view_606, [128, 2048]);  view_606 = None
    permute_360: "f32[2048, 128]" = torch.ops.aten.permute.default(view_610, [1, 0])
    mm_107: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_360, view_464);  permute_360 = None
    permute_361: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    mm_108: "f32[128, 2048]" = torch.ops.aten.mm.default(view_610, permute_362);  view_610 = permute_362 = None
    view_611: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_108, [1, 128, 2048]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_217: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_609, view_611);  view_609 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_363: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_612: "f32[128, 2048]" = torch.ops.aten.view.default(view_607, [128, 2048]);  view_607 = None
    permute_364: "f32[2048, 128]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_109: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_364, view_464);  permute_364 = view_464 = None
    permute_365: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    mm_110: "f32[128, 2048]" = torch.ops.aten.mm.default(view_612, permute_366);  view_612 = permute_366 = None
    view_613: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_110, [1, 128, 2048]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_218: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_217, view_613);  add_217 = view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_367: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_272: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_218, primals_276);  primals_276 = None
    mul_273: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_272, 2048)
    sum_65: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_272, mul_168);  mul_272 = None
    sum_66: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, sum_66);  sum_66 = None
    sub_101: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_273, sum_65);  mul_273 = sum_65 = None
    sub_102: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_101, mul_275);  sub_101 = mul_275 = None
    mul_276: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_32, sub_102);  div_32 = sub_102 = None
    mul_277: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_218, mul_168);  mul_168 = None
    sum_67: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_68: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_219: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_214, mul_276);  add_214 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_614: "f32[128, 2048]" = torch.ops.aten.view.default(add_219, [128, 2048])
    mm_111: "f32[128, 8192]" = torch.ops.aten.mm.default(view_614, permute_368);  permute_368 = None
    permute_369: "f32[2048, 128]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_112: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_369, view_462);  permute_369 = view_462 = None
    permute_370: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_69: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[2048]" = torch.ops.aten.view.default(sum_69, [2048]);  sum_69 = None
    permute_371: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_616: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_111, [1, 128, 8192]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_278: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_616, mul_164);  mul_164 = None
    mul_279: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_616, add_167);  view_616 = add_167 = None
    alias_56: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    mul_280: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_56, alias_56);  alias_56 = None
    sub_103: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_280);  mul_280 = None
    mul_281: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_278, sub_103);  mul_278 = sub_103 = None
    mul_282: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_281, 0.7978845608028654);  mul_281 = None
    mul_283: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_282, 0.044715)
    pow_28: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_461, 2.0);  view_461 = None
    mul_284: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
    mul_285: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_220: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_282, mul_285);  mul_282 = mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_286: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_279, 0.5);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_221: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_220, mul_286);  add_220 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_617: "f32[128, 8192]" = torch.ops.aten.view.default(add_221, [128, 8192]);  add_221 = None
    mm_113: "f32[128, 2048]" = torch.ops.aten.mm.default(view_617, permute_372);  permute_372 = None
    permute_373: "f32[8192, 128]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_114: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_373, view_460);  permute_373 = view_460 = None
    permute_374: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_70: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[8192]" = torch.ops.aten.view.default(sum_70, [8192]);  sum_70 = None
    permute_375: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_619: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_113, [1, 128, 2048]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_288: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_619, primals_270);  primals_270 = None
    mul_289: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_288, 2048)
    sum_71: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
    mul_290: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_288, mul_162);  mul_288 = None
    sum_72: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
    mul_291: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, sum_72);  sum_72 = None
    sub_105: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_289, sum_71);  mul_289 = sum_71 = None
    sub_106: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_105, mul_291);  sub_105 = mul_291 = None
    mul_292: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_33, sub_106);  div_33 = sub_106 = None
    mul_293: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_619, mul_162);  mul_162 = None
    sum_73: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
    sum_74: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_619, [0, 1]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_222: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_219, mul_292);  add_219 = mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_620: "f32[128, 2048]" = torch.ops.aten.view.default(add_222, [128, 2048])
    mm_115: "f32[128, 2048]" = torch.ops.aten.mm.default(view_620, permute_376);  permute_376 = None
    permute_377: "f32[2048, 128]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_116: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_377, view_458);  permute_377 = view_458 = None
    permute_378: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_75: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_620, [0], True);  view_620 = None
    view_621: "f32[2048]" = torch.ops.aten.view.default(sum_75, [2048]);  sum_75 = None
    permute_379: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_622: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_115, [1, 128, 2048]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_623: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_622, [1, 128, 16, 128]);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_380: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_624: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_380, [16, 128, 128]);  permute_380 = None
    bmm_60: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_381, view_624);  permute_381 = None
    bmm_61: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_624, permute_382);  view_624 = permute_382 = None
    view_625: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_60, [1, 16, 128, 128]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_223: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_44, view_625);  tangents_44 = view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_626: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_294: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_626, alias_57);  view_626 = None
    sum_76: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [-1], True)
    mul_295: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_57, sum_76);  alias_57 = sum_76 = None
    sub_107: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_31: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_84, sub_107, full_default_25);  slice_84 = sub_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_627: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_31, [16, 128, 128]);  where_31 = None
    bmm_62: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_383, view_627);  permute_383 = None
    bmm_63: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_627, permute_384);  view_627 = permute_384 = None
    view_628: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_62, [1, 16, 128, 128]);  bmm_62 = None
    view_629: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_63, [1, 16, 128, 128]);  bmm_63 = None
    permute_385: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_628, [0, 1, 3, 2]);  view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_224: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_43, permute_385);  tangents_43 = permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_386: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_223, [0, 2, 1, 3]);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_106: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_630: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_106, [1, 128, 2048]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_387: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_224, [0, 2, 1, 3]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_107: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_631: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_107, [1, 128, 2048]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_388: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_108: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    view_632: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_108, [1, 128, 2048]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_633: "f32[128, 2048]" = torch.ops.aten.view.default(view_630, [128, 2048]);  view_630 = None
    permute_389: "f32[2048, 128]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_117: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_389, view_442);  permute_389 = None
    permute_390: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    mm_118: "f32[128, 2048]" = torch.ops.aten.mm.default(view_633, permute_391);  view_633 = permute_391 = None
    view_634: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_118, [1, 128, 2048]);  mm_118 = None
    permute_392: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_635: "f32[128, 2048]" = torch.ops.aten.view.default(view_631, [128, 2048]);  view_631 = None
    permute_393: "f32[2048, 128]" = torch.ops.aten.permute.default(view_635, [1, 0])
    mm_119: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_393, view_442);  permute_393 = None
    permute_394: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    mm_120: "f32[128, 2048]" = torch.ops.aten.mm.default(view_635, permute_395);  view_635 = permute_395 = None
    view_636: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_120, [1, 128, 2048]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_225: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_634, view_636);  view_634 = view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_396: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_637: "f32[128, 2048]" = torch.ops.aten.view.default(view_632, [128, 2048]);  view_632 = None
    permute_397: "f32[2048, 128]" = torch.ops.aten.permute.default(view_637, [1, 0])
    mm_121: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_397, view_442);  permute_397 = view_442 = None
    permute_398: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    mm_122: "f32[128, 2048]" = torch.ops.aten.mm.default(view_637, permute_399);  view_637 = permute_399 = None
    view_638: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_122, [1, 128, 2048]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_226: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_225, view_638);  add_225 = view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_400: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_297: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_226, primals_263);  primals_263 = None
    mul_298: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_297, 2048)
    sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_297, mul_160);  mul_297 = None
    sum_78: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, sum_78);  sum_78 = None
    sub_109: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_298, sum_77);  mul_298 = sum_77 = None
    sub_110: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_109, mul_300);  sub_109 = mul_300 = None
    mul_301: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_34, sub_110);  div_34 = sub_110 = None
    mul_302: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_226, mul_160);  mul_160 = None
    sum_79: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_80: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_227: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_222, mul_301);  add_222 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_639: "f32[128, 2048]" = torch.ops.aten.view.default(add_227, [128, 2048])
    mm_123: "f32[128, 8192]" = torch.ops.aten.mm.default(view_639, permute_401);  permute_401 = None
    permute_402: "f32[2048, 128]" = torch.ops.aten.permute.default(view_639, [1, 0])
    mm_124: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_402, view_440);  permute_402 = view_440 = None
    permute_403: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_81: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[2048]" = torch.ops.aten.view.default(sum_81, [2048]);  sum_81 = None
    permute_404: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    view_641: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_123, [1, 128, 8192]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_303: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_641, mul_156);  mul_156 = None
    mul_304: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_641, add_159);  view_641 = add_159 = None
    alias_58: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    mul_305: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_58, alias_58);  alias_58 = None
    sub_111: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_305);  mul_305 = None
    mul_306: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_303, sub_111);  mul_303 = sub_111 = None
    mul_307: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_306, 0.7978845608028654);  mul_306 = None
    mul_308: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_307, 0.044715)
    pow_29: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_439, 2.0);  view_439 = None
    mul_309: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
    mul_310: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_228: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_307, mul_310);  mul_307 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_311: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_304, 0.5);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_229: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_228, mul_311);  add_228 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_642: "f32[128, 8192]" = torch.ops.aten.view.default(add_229, [128, 8192]);  add_229 = None
    mm_125: "f32[128, 2048]" = torch.ops.aten.mm.default(view_642, permute_405);  permute_405 = None
    permute_406: "f32[8192, 128]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_126: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_406, view_438);  permute_406 = view_438 = None
    permute_407: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_82: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[8192]" = torch.ops.aten.view.default(sum_82, [8192]);  sum_82 = None
    permute_408: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_644: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_125, [1, 128, 2048]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_313: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_644, primals_257);  primals_257 = None
    mul_314: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_313, 2048)
    sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_313, mul_154);  mul_313 = None
    sum_84: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, sum_84);  sum_84 = None
    sub_113: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_314, sum_83);  mul_314 = sum_83 = None
    sub_114: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_113, mul_316);  sub_113 = mul_316 = None
    mul_317: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_35, sub_114);  div_35 = sub_114 = None
    mul_318: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_644, mul_154);  mul_154 = None
    sum_85: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_86: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_644, [0, 1]);  view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_230: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_227, mul_317);  add_227 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_645: "f32[128, 2048]" = torch.ops.aten.view.default(add_230, [128, 2048])
    mm_127: "f32[128, 2048]" = torch.ops.aten.mm.default(view_645, permute_409);  permute_409 = None
    permute_410: "f32[2048, 128]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_128: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_410, view_436);  permute_410 = view_436 = None
    permute_411: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_87: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[2048]" = torch.ops.aten.view.default(sum_87, [2048]);  sum_87 = None
    permute_412: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_647: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_127, [1, 128, 2048]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_648: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_647, [1, 128, 16, 128]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_413: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_649: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_413, [16, 128, 128]);  permute_413 = None
    bmm_64: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_414, view_649);  permute_414 = None
    bmm_65: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_649, permute_415);  view_649 = permute_415 = None
    view_650: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_64, [1, 16, 128, 128]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_231: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_42, view_650);  tangents_42 = view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_651: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_319: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_651, alias_59);  view_651 = None
    sum_88: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
    mul_320: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_59, sum_88);  alias_59 = sum_88 = None
    sub_115: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_32: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_80, sub_115, full_default_25);  slice_80 = sub_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_652: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_32, [16, 128, 128]);  where_32 = None
    bmm_66: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_416, view_652);  permute_416 = None
    bmm_67: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_652, permute_417);  view_652 = permute_417 = None
    view_653: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_66, [1, 16, 128, 128]);  bmm_66 = None
    view_654: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_67, [1, 16, 128, 128]);  bmm_67 = None
    permute_418: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_653, [0, 1, 3, 2]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_232: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_41, permute_418);  tangents_41 = permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_419: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_231, [0, 2, 1, 3]);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_109: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_655: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_109, [1, 128, 2048]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_420: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_232, [0, 2, 1, 3]);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_110: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_656: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_110, [1, 128, 2048]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_421: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_111: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_657: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_111, [1, 128, 2048]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_658: "f32[128, 2048]" = torch.ops.aten.view.default(view_655, [128, 2048]);  view_655 = None
    permute_422: "f32[2048, 128]" = torch.ops.aten.permute.default(view_658, [1, 0])
    mm_129: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_422, view_420);  permute_422 = None
    permute_423: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    mm_130: "f32[128, 2048]" = torch.ops.aten.mm.default(view_658, permute_424);  view_658 = permute_424 = None
    view_659: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_130, [1, 128, 2048]);  mm_130 = None
    permute_425: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_660: "f32[128, 2048]" = torch.ops.aten.view.default(view_656, [128, 2048]);  view_656 = None
    permute_426: "f32[2048, 128]" = torch.ops.aten.permute.default(view_660, [1, 0])
    mm_131: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_426, view_420);  permute_426 = None
    permute_427: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    mm_132: "f32[128, 2048]" = torch.ops.aten.mm.default(view_660, permute_428);  view_660 = permute_428 = None
    view_661: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_132, [1, 128, 2048]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_233: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_659, view_661);  view_659 = view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_429: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_662: "f32[128, 2048]" = torch.ops.aten.view.default(view_657, [128, 2048]);  view_657 = None
    permute_430: "f32[2048, 128]" = torch.ops.aten.permute.default(view_662, [1, 0])
    mm_133: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_430, view_420);  permute_430 = view_420 = None
    permute_431: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    mm_134: "f32[128, 2048]" = torch.ops.aten.mm.default(view_662, permute_432);  view_662 = permute_432 = None
    view_663: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_134, [1, 128, 2048]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_234: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_233, view_663);  add_233 = view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_433: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_322: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_234, primals_250);  primals_250 = None
    mul_323: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_322, 2048)
    sum_89: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True)
    mul_324: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_322, mul_152);  mul_322 = None
    sum_90: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
    mul_325: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, sum_90);  sum_90 = None
    sub_117: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_323, sum_89);  mul_323 = sum_89 = None
    sub_118: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_117, mul_325);  sub_117 = mul_325 = None
    mul_326: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_36, sub_118);  div_36 = sub_118 = None
    mul_327: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_234, mul_152);  mul_152 = None
    sum_91: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1]);  mul_327 = None
    sum_92: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_234, [0, 1]);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_235: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_230, mul_326);  add_230 = mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_664: "f32[128, 2048]" = torch.ops.aten.view.default(add_235, [128, 2048])
    mm_135: "f32[128, 8192]" = torch.ops.aten.mm.default(view_664, permute_434);  permute_434 = None
    permute_435: "f32[2048, 128]" = torch.ops.aten.permute.default(view_664, [1, 0])
    mm_136: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_435, view_418);  permute_435 = view_418 = None
    permute_436: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_93: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
    view_665: "f32[2048]" = torch.ops.aten.view.default(sum_93, [2048]);  sum_93 = None
    permute_437: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_666: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_135, [1, 128, 8192]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_328: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_666, mul_148);  mul_148 = None
    mul_329: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_666, add_151);  view_666 = add_151 = None
    alias_60: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    mul_330: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_60, alias_60);  alias_60 = None
    sub_119: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_330);  mul_330 = None
    mul_331: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_328, sub_119);  mul_328 = sub_119 = None
    mul_332: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_331, 0.7978845608028654);  mul_331 = None
    mul_333: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_332, 0.044715)
    pow_30: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 2.0);  view_417 = None
    mul_334: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
    mul_335: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_236: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_332, mul_335);  mul_332 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_336: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_329, 0.5);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_237: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_236, mul_336);  add_236 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_667: "f32[128, 8192]" = torch.ops.aten.view.default(add_237, [128, 8192]);  add_237 = None
    mm_137: "f32[128, 2048]" = torch.ops.aten.mm.default(view_667, permute_438);  permute_438 = None
    permute_439: "f32[8192, 128]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_138: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_439, view_416);  permute_439 = view_416 = None
    permute_440: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_94: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
    view_668: "f32[8192]" = torch.ops.aten.view.default(sum_94, [8192]);  sum_94 = None
    permute_441: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_669: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_137, [1, 128, 2048]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_338: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_669, primals_244);  primals_244 = None
    mul_339: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_338, 2048)
    sum_95: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_338, mul_146);  mul_338 = None
    sum_96: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, sum_96);  sum_96 = None
    sub_121: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_339, sum_95);  mul_339 = sum_95 = None
    sub_122: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_121, mul_341);  sub_121 = mul_341 = None
    mul_342: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_37, sub_122);  div_37 = sub_122 = None
    mul_343: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_669, mul_146);  mul_146 = None
    sum_97: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_98: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_669, [0, 1]);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_238: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_235, mul_342);  add_235 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_670: "f32[128, 2048]" = torch.ops.aten.view.default(add_238, [128, 2048])
    mm_139: "f32[128, 2048]" = torch.ops.aten.mm.default(view_670, permute_442);  permute_442 = None
    permute_443: "f32[2048, 128]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_140: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_443, view_414);  permute_443 = view_414 = None
    permute_444: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_99: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[2048]" = torch.ops.aten.view.default(sum_99, [2048]);  sum_99 = None
    permute_445: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_672: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_139, [1, 128, 2048]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_673: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_672, [1, 128, 16, 128]);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_674: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_446, [16, 128, 128]);  permute_446 = None
    bmm_68: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_447, view_674);  permute_447 = None
    bmm_69: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_674, permute_448);  view_674 = permute_448 = None
    view_675: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_68, [1, 16, 128, 128]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_239: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_40, view_675);  tangents_40 = view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_676: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_344: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_676, alias_61);  view_676 = None
    sum_100: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_61, sum_100);  alias_61 = sum_100 = None
    sub_123: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_33: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_76, sub_123, full_default_25);  slice_76 = sub_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_677: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_33, [16, 128, 128]);  where_33 = None
    bmm_70: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_449, view_677);  permute_449 = None
    bmm_71: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_677, permute_450);  view_677 = permute_450 = None
    view_678: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_70, [1, 16, 128, 128]);  bmm_70 = None
    view_679: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_71, [1, 16, 128, 128]);  bmm_71 = None
    permute_451: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_678, [0, 1, 3, 2]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_240: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_39, permute_451);  tangents_39 = permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_452: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_239, [0, 2, 1, 3]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_112: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_680: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_112, [1, 128, 2048]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_453: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_240, [0, 2, 1, 3]);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_113: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_681: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_113, [1, 128, 2048]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_454: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_679, [0, 2, 1, 3]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_114: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_682: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_114, [1, 128, 2048]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_683: "f32[128, 2048]" = torch.ops.aten.view.default(view_680, [128, 2048]);  view_680 = None
    permute_455: "f32[2048, 128]" = torch.ops.aten.permute.default(view_683, [1, 0])
    mm_141: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_455, view_398);  permute_455 = None
    permute_456: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    mm_142: "f32[128, 2048]" = torch.ops.aten.mm.default(view_683, permute_457);  view_683 = permute_457 = None
    view_684: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_142, [1, 128, 2048]);  mm_142 = None
    permute_458: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_685: "f32[128, 2048]" = torch.ops.aten.view.default(view_681, [128, 2048]);  view_681 = None
    permute_459: "f32[2048, 128]" = torch.ops.aten.permute.default(view_685, [1, 0])
    mm_143: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_459, view_398);  permute_459 = None
    permute_460: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    mm_144: "f32[128, 2048]" = torch.ops.aten.mm.default(view_685, permute_461);  view_685 = permute_461 = None
    view_686: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_144, [1, 128, 2048]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_241: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_684, view_686);  view_684 = view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_462: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_687: "f32[128, 2048]" = torch.ops.aten.view.default(view_682, [128, 2048]);  view_682 = None
    permute_463: "f32[2048, 128]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_145: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_463, view_398);  permute_463 = view_398 = None
    permute_464: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    mm_146: "f32[128, 2048]" = torch.ops.aten.mm.default(view_687, permute_465);  view_687 = permute_465 = None
    view_688: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_146, [1, 128, 2048]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_242: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_241, view_688);  add_241 = view_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_466: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_347: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_242, primals_237);  primals_237 = None
    mul_348: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_347, 2048)
    sum_101: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_347, mul_144);  mul_347 = None
    sum_102: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, sum_102);  sum_102 = None
    sub_125: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_348, sum_101);  mul_348 = sum_101 = None
    sub_126: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_125, mul_350);  sub_125 = mul_350 = None
    mul_351: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_38, sub_126);  div_38 = sub_126 = None
    mul_352: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_242, mul_144);  mul_144 = None
    sum_103: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_104: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_243: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_238, mul_351);  add_238 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_689: "f32[128, 2048]" = torch.ops.aten.view.default(add_243, [128, 2048])
    mm_147: "f32[128, 8192]" = torch.ops.aten.mm.default(view_689, permute_467);  permute_467 = None
    permute_468: "f32[2048, 128]" = torch.ops.aten.permute.default(view_689, [1, 0])
    mm_148: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_468, view_396);  permute_468 = view_396 = None
    permute_469: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_105: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_689, [0], True);  view_689 = None
    view_690: "f32[2048]" = torch.ops.aten.view.default(sum_105, [2048]);  sum_105 = None
    permute_470: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_691: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_147, [1, 128, 8192]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_353: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_691, mul_140);  mul_140 = None
    mul_354: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_691, add_143);  view_691 = add_143 = None
    alias_62: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_355: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_62, alias_62);  alias_62 = None
    sub_127: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_355);  mul_355 = None
    mul_356: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_353, sub_127);  mul_353 = sub_127 = None
    mul_357: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_356, 0.7978845608028654);  mul_356 = None
    mul_358: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_357, 0.044715)
    pow_31: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_395, 2.0);  view_395 = None
    mul_359: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
    mul_360: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_244: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_357, mul_360);  mul_357 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_361: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_354, 0.5);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_245: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_244, mul_361);  add_244 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_692: "f32[128, 8192]" = torch.ops.aten.view.default(add_245, [128, 8192]);  add_245 = None
    mm_149: "f32[128, 2048]" = torch.ops.aten.mm.default(view_692, permute_471);  permute_471 = None
    permute_472: "f32[8192, 128]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_150: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_472, view_394);  permute_472 = view_394 = None
    permute_473: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_106: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[8192]" = torch.ops.aten.view.default(sum_106, [8192]);  sum_106 = None
    permute_474: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_694: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_149, [1, 128, 2048]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_363: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_694, primals_231);  primals_231 = None
    mul_364: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_363, 2048)
    sum_107: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
    mul_365: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_363, mul_138);  mul_363 = None
    sum_108: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    mul_366: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, sum_108);  sum_108 = None
    sub_129: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_364, sum_107);  mul_364 = sum_107 = None
    sub_130: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_129, mul_366);  sub_129 = mul_366 = None
    mul_367: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_39, sub_130);  div_39 = sub_130 = None
    mul_368: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_694, mul_138);  mul_138 = None
    sum_109: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
    sum_110: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_694, [0, 1]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_246: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_243, mul_367);  add_243 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_695: "f32[128, 2048]" = torch.ops.aten.view.default(add_246, [128, 2048])
    mm_151: "f32[128, 2048]" = torch.ops.aten.mm.default(view_695, permute_475);  permute_475 = None
    permute_476: "f32[2048, 128]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_152: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_476, view_392);  permute_476 = view_392 = None
    permute_477: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_111: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[2048]" = torch.ops.aten.view.default(sum_111, [2048]);  sum_111 = None
    permute_478: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_697: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_151, [1, 128, 2048]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_698: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_697, [1, 128, 16, 128]);  view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_479: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_699: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_479, [16, 128, 128]);  permute_479 = None
    bmm_72: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_480, view_699);  permute_480 = None
    bmm_73: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_699, permute_481);  view_699 = permute_481 = None
    view_700: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_72, [1, 16, 128, 128]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_247: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_38, view_700);  tangents_38 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_701: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_369: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_701, alias_63);  view_701 = None
    sum_112: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [-1], True)
    mul_370: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_63, sum_112);  alias_63 = sum_112 = None
    sub_131: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_34: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_72, sub_131, full_default_25);  slice_72 = sub_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_702: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_34, [16, 128, 128]);  where_34 = None
    bmm_74: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_482, view_702);  permute_482 = None
    bmm_75: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_702, permute_483);  view_702 = permute_483 = None
    view_703: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_74, [1, 16, 128, 128]);  bmm_74 = None
    view_704: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_75, [1, 16, 128, 128]);  bmm_75 = None
    permute_484: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_703, [0, 1, 3, 2]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_248: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_37, permute_484);  tangents_37 = permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_485: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_247, [0, 2, 1, 3]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_115: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_705: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_115, [1, 128, 2048]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_486: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_248, [0, 2, 1, 3]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_116: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_706: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_116, [1, 128, 2048]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_487: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_117: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_707: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_117, [1, 128, 2048]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_708: "f32[128, 2048]" = torch.ops.aten.view.default(view_705, [128, 2048]);  view_705 = None
    permute_488: "f32[2048, 128]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_153: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_488, view_376);  permute_488 = None
    permute_489: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    mm_154: "f32[128, 2048]" = torch.ops.aten.mm.default(view_708, permute_490);  view_708 = permute_490 = None
    view_709: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_154, [1, 128, 2048]);  mm_154 = None
    permute_491: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_710: "f32[128, 2048]" = torch.ops.aten.view.default(view_706, [128, 2048]);  view_706 = None
    permute_492: "f32[2048, 128]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_155: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_492, view_376);  permute_492 = None
    permute_493: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    mm_156: "f32[128, 2048]" = torch.ops.aten.mm.default(view_710, permute_494);  view_710 = permute_494 = None
    view_711: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_156, [1, 128, 2048]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_249: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_709, view_711);  view_709 = view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_495: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_712: "f32[128, 2048]" = torch.ops.aten.view.default(view_707, [128, 2048]);  view_707 = None
    permute_496: "f32[2048, 128]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_157: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_496, view_376);  permute_496 = view_376 = None
    permute_497: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    mm_158: "f32[128, 2048]" = torch.ops.aten.mm.default(view_712, permute_498);  view_712 = permute_498 = None
    view_713: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_158, [1, 128, 2048]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_250: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_249, view_713);  add_249 = view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_499: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_372: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_250, primals_224);  primals_224 = None
    mul_373: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_372, 2048)
    sum_113: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
    mul_374: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_372, mul_136);  mul_372 = None
    sum_114: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
    mul_375: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, sum_114);  sum_114 = None
    sub_133: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_373, sum_113);  mul_373 = sum_113 = None
    sub_134: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_133, mul_375);  sub_133 = mul_375 = None
    mul_376: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_40, sub_134);  div_40 = sub_134 = None
    mul_377: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_250, mul_136);  mul_136 = None
    sum_115: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
    sum_116: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_250, [0, 1]);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_251: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_246, mul_376);  add_246 = mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_714: "f32[128, 2048]" = torch.ops.aten.view.default(add_251, [128, 2048])
    mm_159: "f32[128, 8192]" = torch.ops.aten.mm.default(view_714, permute_500);  permute_500 = None
    permute_501: "f32[2048, 128]" = torch.ops.aten.permute.default(view_714, [1, 0])
    mm_160: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_501, view_374);  permute_501 = view_374 = None
    permute_502: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_117: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_714, [0], True);  view_714 = None
    view_715: "f32[2048]" = torch.ops.aten.view.default(sum_117, [2048]);  sum_117 = None
    permute_503: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_716: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_159, [1, 128, 8192]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_378: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_716, mul_132);  mul_132 = None
    mul_379: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_716, add_135);  view_716 = add_135 = None
    alias_64: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_380: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_64, alias_64);  alias_64 = None
    sub_135: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_380);  mul_380 = None
    mul_381: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_378, sub_135);  mul_378 = sub_135 = None
    mul_382: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_381, 0.7978845608028654);  mul_381 = None
    mul_383: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_382, 0.044715)
    pow_32: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_373, 2.0);  view_373 = None
    mul_384: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
    mul_385: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_252: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_382, mul_385);  mul_382 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_386: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_379, 0.5);  mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_253: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_252, mul_386);  add_252 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_717: "f32[128, 8192]" = torch.ops.aten.view.default(add_253, [128, 8192]);  add_253 = None
    mm_161: "f32[128, 2048]" = torch.ops.aten.mm.default(view_717, permute_504);  permute_504 = None
    permute_505: "f32[8192, 128]" = torch.ops.aten.permute.default(view_717, [1, 0])
    mm_162: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_505, view_372);  permute_505 = view_372 = None
    permute_506: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_118: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_717, [0], True);  view_717 = None
    view_718: "f32[8192]" = torch.ops.aten.view.default(sum_118, [8192]);  sum_118 = None
    permute_507: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_719: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_161, [1, 128, 2048]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_388: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_719, primals_218);  primals_218 = None
    mul_389: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_388, 2048)
    sum_119: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_388, mul_130);  mul_388 = None
    sum_120: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, sum_120);  sum_120 = None
    sub_137: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_389, sum_119);  mul_389 = sum_119 = None
    sub_138: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_137, mul_391);  sub_137 = mul_391 = None
    mul_392: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_41, sub_138);  div_41 = sub_138 = None
    mul_393: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_719, mul_130);  mul_130 = None
    sum_121: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_122: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_719, [0, 1]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_254: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_251, mul_392);  add_251 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_720: "f32[128, 2048]" = torch.ops.aten.view.default(add_254, [128, 2048])
    mm_163: "f32[128, 2048]" = torch.ops.aten.mm.default(view_720, permute_508);  permute_508 = None
    permute_509: "f32[2048, 128]" = torch.ops.aten.permute.default(view_720, [1, 0])
    mm_164: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_509, view_370);  permute_509 = view_370 = None
    permute_510: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_123: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_720, [0], True);  view_720 = None
    view_721: "f32[2048]" = torch.ops.aten.view.default(sum_123, [2048]);  sum_123 = None
    permute_511: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_722: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_163, [1, 128, 2048]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_723: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_722, [1, 128, 16, 128]);  view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_512: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_724: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_512, [16, 128, 128]);  permute_512 = None
    bmm_76: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_513, view_724);  permute_513 = None
    bmm_77: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_724, permute_514);  view_724 = permute_514 = None
    view_725: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_76, [1, 16, 128, 128]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_255: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_36, view_725);  tangents_36 = view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_726: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_394: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_726, alias_65);  view_726 = None
    sum_124: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1], True)
    mul_395: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_65, sum_124);  alias_65 = sum_124 = None
    sub_139: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_35: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_68, sub_139, full_default_25);  slice_68 = sub_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_727: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_35, [16, 128, 128]);  where_35 = None
    bmm_78: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_515, view_727);  permute_515 = None
    bmm_79: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_727, permute_516);  view_727 = permute_516 = None
    view_728: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_78, [1, 16, 128, 128]);  bmm_78 = None
    view_729: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_79, [1, 16, 128, 128]);  bmm_79 = None
    permute_517: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_728, [0, 1, 3, 2]);  view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_256: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_35, permute_517);  tangents_35 = permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_518: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_255, [0, 2, 1, 3]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_118: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_730: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_118, [1, 128, 2048]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_519: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_256, [0, 2, 1, 3]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_119: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_731: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_119, [1, 128, 2048]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_520: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_729, [0, 2, 1, 3]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_120: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_732: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_120, [1, 128, 2048]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_733: "f32[128, 2048]" = torch.ops.aten.view.default(view_730, [128, 2048]);  view_730 = None
    permute_521: "f32[2048, 128]" = torch.ops.aten.permute.default(view_733, [1, 0])
    mm_165: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_521, view_354);  permute_521 = None
    permute_522: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    mm_166: "f32[128, 2048]" = torch.ops.aten.mm.default(view_733, permute_523);  view_733 = permute_523 = None
    view_734: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_166, [1, 128, 2048]);  mm_166 = None
    permute_524: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_735: "f32[128, 2048]" = torch.ops.aten.view.default(view_731, [128, 2048]);  view_731 = None
    permute_525: "f32[2048, 128]" = torch.ops.aten.permute.default(view_735, [1, 0])
    mm_167: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_525, view_354);  permute_525 = None
    permute_526: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    mm_168: "f32[128, 2048]" = torch.ops.aten.mm.default(view_735, permute_527);  view_735 = permute_527 = None
    view_736: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_168, [1, 128, 2048]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_257: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_734, view_736);  view_734 = view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_528: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_737: "f32[128, 2048]" = torch.ops.aten.view.default(view_732, [128, 2048]);  view_732 = None
    permute_529: "f32[2048, 128]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_169: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_529, view_354);  permute_529 = view_354 = None
    permute_530: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    mm_170: "f32[128, 2048]" = torch.ops.aten.mm.default(view_737, permute_531);  view_737 = permute_531 = None
    view_738: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_170, [1, 128, 2048]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_258: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_257, view_738);  add_257 = view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_532: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_397: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_258, primals_211);  primals_211 = None
    mul_398: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_397, 2048)
    sum_125: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_397, mul_128);  mul_397 = None
    sum_126: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, sum_126);  sum_126 = None
    sub_141: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_398, sum_125);  mul_398 = sum_125 = None
    sub_142: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_141, mul_400);  sub_141 = mul_400 = None
    mul_401: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_42, sub_142);  div_42 = sub_142 = None
    mul_402: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_258, mul_128);  mul_128 = None
    sum_127: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_128: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_258, [0, 1]);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_259: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_254, mul_401);  add_254 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_739: "f32[128, 2048]" = torch.ops.aten.view.default(add_259, [128, 2048])
    mm_171: "f32[128, 8192]" = torch.ops.aten.mm.default(view_739, permute_533);  permute_533 = None
    permute_534: "f32[2048, 128]" = torch.ops.aten.permute.default(view_739, [1, 0])
    mm_172: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_534, view_352);  permute_534 = view_352 = None
    permute_535: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_129: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_739, [0], True);  view_739 = None
    view_740: "f32[2048]" = torch.ops.aten.view.default(sum_129, [2048]);  sum_129 = None
    permute_536: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_535, [1, 0]);  permute_535 = None
    view_741: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_171, [1, 128, 8192]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_403: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_741, mul_124);  mul_124 = None
    mul_404: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_741, add_127);  view_741 = add_127 = None
    alias_66: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    mul_405: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_66, alias_66);  alias_66 = None
    sub_143: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
    mul_406: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_403, sub_143);  mul_403 = sub_143 = None
    mul_407: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
    mul_408: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_407, 0.044715)
    pow_33: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_351, 2.0);  view_351 = None
    mul_409: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
    mul_410: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_260: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_407, mul_410);  mul_407 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_411: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_404, 0.5);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_261: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_260, mul_411);  add_260 = mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_742: "f32[128, 8192]" = torch.ops.aten.view.default(add_261, [128, 8192]);  add_261 = None
    mm_173: "f32[128, 2048]" = torch.ops.aten.mm.default(view_742, permute_537);  permute_537 = None
    permute_538: "f32[8192, 128]" = torch.ops.aten.permute.default(view_742, [1, 0])
    mm_174: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_538, view_350);  permute_538 = view_350 = None
    permute_539: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_130: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_742, [0], True);  view_742 = None
    view_743: "f32[8192]" = torch.ops.aten.view.default(sum_130, [8192]);  sum_130 = None
    permute_540: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    view_744: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_173, [1, 128, 2048]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_413: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_744, primals_205);  primals_205 = None
    mul_414: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_413, 2048)
    sum_131: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
    mul_415: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_413, mul_122);  mul_413 = None
    sum_132: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
    mul_416: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, sum_132);  sum_132 = None
    sub_145: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_414, sum_131);  mul_414 = sum_131 = None
    sub_146: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_145, mul_416);  sub_145 = mul_416 = None
    mul_417: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_43, sub_146);  div_43 = sub_146 = None
    mul_418: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_744, mul_122);  mul_122 = None
    sum_133: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
    sum_134: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_744, [0, 1]);  view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_262: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_259, mul_417);  add_259 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_745: "f32[128, 2048]" = torch.ops.aten.view.default(add_262, [128, 2048])
    mm_175: "f32[128, 2048]" = torch.ops.aten.mm.default(view_745, permute_541);  permute_541 = None
    permute_542: "f32[2048, 128]" = torch.ops.aten.permute.default(view_745, [1, 0])
    mm_176: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_542, view_348);  permute_542 = view_348 = None
    permute_543: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_135: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_745, [0], True);  view_745 = None
    view_746: "f32[2048]" = torch.ops.aten.view.default(sum_135, [2048]);  sum_135 = None
    permute_544: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    view_747: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_175, [1, 128, 2048]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_748: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_747, [1, 128, 16, 128]);  view_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_545: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_749: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_545, [16, 128, 128]);  permute_545 = None
    bmm_80: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_546, view_749);  permute_546 = None
    bmm_81: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_749, permute_547);  view_749 = permute_547 = None
    view_750: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_80, [1, 16, 128, 128]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_263: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_34, view_750);  tangents_34 = view_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_751: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_419: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_751, alias_67);  view_751 = None
    sum_136: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
    mul_420: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_67, sum_136);  alias_67 = sum_136 = None
    sub_147: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_36: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_64, sub_147, full_default_25);  slice_64 = sub_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_752: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_36, [16, 128, 128]);  where_36 = None
    bmm_82: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_548, view_752);  permute_548 = None
    bmm_83: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_752, permute_549);  view_752 = permute_549 = None
    view_753: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_82, [1, 16, 128, 128]);  bmm_82 = None
    view_754: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_83, [1, 16, 128, 128]);  bmm_83 = None
    permute_550: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_753, [0, 1, 3, 2]);  view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_264: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_33, permute_550);  tangents_33 = permute_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_551: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_263, [0, 2, 1, 3]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_121: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_755: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_121, [1, 128, 2048]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_552: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_264, [0, 2, 1, 3]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_122: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_756: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_122, [1, 128, 2048]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_553: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_754, [0, 2, 1, 3]);  view_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_123: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
    view_757: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_123, [1, 128, 2048]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_758: "f32[128, 2048]" = torch.ops.aten.view.default(view_755, [128, 2048]);  view_755 = None
    permute_554: "f32[2048, 128]" = torch.ops.aten.permute.default(view_758, [1, 0])
    mm_177: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_554, view_332);  permute_554 = None
    permute_555: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    mm_178: "f32[128, 2048]" = torch.ops.aten.mm.default(view_758, permute_556);  view_758 = permute_556 = None
    view_759: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_178, [1, 128, 2048]);  mm_178 = None
    permute_557: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_555, [1, 0]);  permute_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_760: "f32[128, 2048]" = torch.ops.aten.view.default(view_756, [128, 2048]);  view_756 = None
    permute_558: "f32[2048, 128]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_179: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_558, view_332);  permute_558 = None
    permute_559: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    mm_180: "f32[128, 2048]" = torch.ops.aten.mm.default(view_760, permute_560);  view_760 = permute_560 = None
    view_761: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_180, [1, 128, 2048]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_265: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_759, view_761);  view_759 = view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_561: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_559, [1, 0]);  permute_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_762: "f32[128, 2048]" = torch.ops.aten.view.default(view_757, [128, 2048]);  view_757 = None
    permute_562: "f32[2048, 128]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_181: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_562, view_332);  permute_562 = view_332 = None
    permute_563: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    mm_182: "f32[128, 2048]" = torch.ops.aten.mm.default(view_762, permute_564);  view_762 = permute_564 = None
    view_763: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_182, [1, 128, 2048]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_266: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_265, view_763);  add_265 = view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_565: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_563, [1, 0]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_422: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_266, primals_198);  primals_198 = None
    mul_423: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_422, 2048)
    sum_137: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
    mul_424: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_422, mul_120);  mul_422 = None
    sum_138: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
    mul_425: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, sum_138);  sum_138 = None
    sub_149: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_423, sum_137);  mul_423 = sum_137 = None
    sub_150: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_149, mul_425);  sub_149 = mul_425 = None
    mul_426: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_44, sub_150);  div_44 = sub_150 = None
    mul_427: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_266, mul_120);  mul_120 = None
    sum_139: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
    sum_140: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_267: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_262, mul_426);  add_262 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_764: "f32[128, 2048]" = torch.ops.aten.view.default(add_267, [128, 2048])
    mm_183: "f32[128, 8192]" = torch.ops.aten.mm.default(view_764, permute_566);  permute_566 = None
    permute_567: "f32[2048, 128]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_184: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_567, view_330);  permute_567 = view_330 = None
    permute_568: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_141: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
    view_765: "f32[2048]" = torch.ops.aten.view.default(sum_141, [2048]);  sum_141 = None
    permute_569: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    view_766: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_183, [1, 128, 8192]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_428: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_766, mul_116);  mul_116 = None
    mul_429: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_766, add_119);  view_766 = add_119 = None
    alias_68: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_430: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_68, alias_68);  alias_68 = None
    sub_151: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_430);  mul_430 = None
    mul_431: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_428, sub_151);  mul_428 = sub_151 = None
    mul_432: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_431, 0.7978845608028654);  mul_431 = None
    mul_433: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_432, 0.044715)
    pow_34: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_329, 2.0);  view_329 = None
    mul_434: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
    mul_435: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_268: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_432, mul_435);  mul_432 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_436: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_429, 0.5);  mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_269: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_268, mul_436);  add_268 = mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_767: "f32[128, 8192]" = torch.ops.aten.view.default(add_269, [128, 8192]);  add_269 = None
    mm_185: "f32[128, 2048]" = torch.ops.aten.mm.default(view_767, permute_570);  permute_570 = None
    permute_571: "f32[8192, 128]" = torch.ops.aten.permute.default(view_767, [1, 0])
    mm_186: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_571, view_328);  permute_571 = view_328 = None
    permute_572: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_142: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
    view_768: "f32[8192]" = torch.ops.aten.view.default(sum_142, [8192]);  sum_142 = None
    permute_573: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    view_769: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_185, [1, 128, 2048]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_438: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_769, primals_192);  primals_192 = None
    mul_439: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_438, 2048)
    sum_143: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True)
    mul_440: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_438, mul_114);  mul_438 = None
    sum_144: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True);  mul_440 = None
    mul_441: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, sum_144);  sum_144 = None
    sub_153: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_439, sum_143);  mul_439 = sum_143 = None
    sub_154: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_153, mul_441);  sub_153 = mul_441 = None
    mul_442: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_45, sub_154);  div_45 = sub_154 = None
    mul_443: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_769, mul_114);  mul_114 = None
    sum_145: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 1]);  mul_443 = None
    sum_146: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_769, [0, 1]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_270: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_267, mul_442);  add_267 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_770: "f32[128, 2048]" = torch.ops.aten.view.default(add_270, [128, 2048])
    mm_187: "f32[128, 2048]" = torch.ops.aten.mm.default(view_770, permute_574);  permute_574 = None
    permute_575: "f32[2048, 128]" = torch.ops.aten.permute.default(view_770, [1, 0])
    mm_188: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_575, view_326);  permute_575 = view_326 = None
    permute_576: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_147: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_770, [0], True);  view_770 = None
    view_771: "f32[2048]" = torch.ops.aten.view.default(sum_147, [2048]);  sum_147 = None
    permute_577: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    view_772: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_187, [1, 128, 2048]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_773: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_772, [1, 128, 16, 128]);  view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_578: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_774: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_578, [16, 128, 128]);  permute_578 = None
    bmm_84: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_579, view_774);  permute_579 = None
    bmm_85: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_774, permute_580);  view_774 = permute_580 = None
    view_775: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_84, [1, 16, 128, 128]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_271: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_32, view_775);  tangents_32 = view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_776: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_444: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_776, alias_69);  view_776 = None
    sum_148: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [-1], True)
    mul_445: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_69, sum_148);  alias_69 = sum_148 = None
    sub_155: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_37: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_60, sub_155, full_default_25);  slice_60 = sub_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_777: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_37, [16, 128, 128]);  where_37 = None
    bmm_86: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_581, view_777);  permute_581 = None
    bmm_87: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_777, permute_582);  view_777 = permute_582 = None
    view_778: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_86, [1, 16, 128, 128]);  bmm_86 = None
    view_779: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_87, [1, 16, 128, 128]);  bmm_87 = None
    permute_583: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_778, [0, 1, 3, 2]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_272: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_31, permute_583);  tangents_31 = permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_584: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_271, [0, 2, 1, 3]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_124: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_780: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_124, [1, 128, 2048]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_585: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_272, [0, 2, 1, 3]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_125: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
    view_781: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_125, [1, 128, 2048]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_586: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_126: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
    view_782: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_126, [1, 128, 2048]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_783: "f32[128, 2048]" = torch.ops.aten.view.default(view_780, [128, 2048]);  view_780 = None
    permute_587: "f32[2048, 128]" = torch.ops.aten.permute.default(view_783, [1, 0])
    mm_189: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_587, view_310);  permute_587 = None
    permute_588: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    mm_190: "f32[128, 2048]" = torch.ops.aten.mm.default(view_783, permute_589);  view_783 = permute_589 = None
    view_784: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_190, [1, 128, 2048]);  mm_190 = None
    permute_590: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_588, [1, 0]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_785: "f32[128, 2048]" = torch.ops.aten.view.default(view_781, [128, 2048]);  view_781 = None
    permute_591: "f32[2048, 128]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_191: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_591, view_310);  permute_591 = None
    permute_592: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    mm_192: "f32[128, 2048]" = torch.ops.aten.mm.default(view_785, permute_593);  view_785 = permute_593 = None
    view_786: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_192, [1, 128, 2048]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_273: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_784, view_786);  view_784 = view_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_594: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_787: "f32[128, 2048]" = torch.ops.aten.view.default(view_782, [128, 2048]);  view_782 = None
    permute_595: "f32[2048, 128]" = torch.ops.aten.permute.default(view_787, [1, 0])
    mm_193: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_595, view_310);  permute_595 = view_310 = None
    permute_596: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    mm_194: "f32[128, 2048]" = torch.ops.aten.mm.default(view_787, permute_597);  view_787 = permute_597 = None
    view_788: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_194, [1, 128, 2048]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_274: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_273, view_788);  add_273 = view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_598: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_596, [1, 0]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_447: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_274, primals_185);  primals_185 = None
    mul_448: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_447, 2048)
    sum_149: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
    mul_449: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_447, mul_112);  mul_447 = None
    sum_150: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
    mul_450: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, sum_150);  sum_150 = None
    sub_157: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_448, sum_149);  mul_448 = sum_149 = None
    sub_158: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_157, mul_450);  sub_157 = mul_450 = None
    mul_451: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_46, sub_158);  div_46 = sub_158 = None
    mul_452: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_274, mul_112);  mul_112 = None
    sum_151: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    sum_152: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_275: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_270, mul_451);  add_270 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_789: "f32[128, 2048]" = torch.ops.aten.view.default(add_275, [128, 2048])
    mm_195: "f32[128, 8192]" = torch.ops.aten.mm.default(view_789, permute_599);  permute_599 = None
    permute_600: "f32[2048, 128]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_196: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_600, view_308);  permute_600 = view_308 = None
    permute_601: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_153: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[2048]" = torch.ops.aten.view.default(sum_153, [2048]);  sum_153 = None
    permute_602: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    view_791: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_195, [1, 128, 8192]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_453: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_791, mul_108);  mul_108 = None
    mul_454: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_791, add_111);  view_791 = add_111 = None
    alias_70: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_455: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_70, alias_70);  alias_70 = None
    sub_159: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_455);  mul_455 = None
    mul_456: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_453, sub_159);  mul_453 = sub_159 = None
    mul_457: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_456, 0.7978845608028654);  mul_456 = None
    mul_458: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_457, 0.044715)
    pow_35: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 2.0);  view_307 = None
    mul_459: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
    mul_460: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_458, mul_459);  mul_458 = mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_276: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_457, mul_460);  mul_457 = mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_461: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_454, 0.5);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_277: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_276, mul_461);  add_276 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_792: "f32[128, 8192]" = torch.ops.aten.view.default(add_277, [128, 8192]);  add_277 = None
    mm_197: "f32[128, 2048]" = torch.ops.aten.mm.default(view_792, permute_603);  permute_603 = None
    permute_604: "f32[8192, 128]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_198: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_604, view_306);  permute_604 = view_306 = None
    permute_605: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_198, [1, 0]);  mm_198 = None
    sum_154: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_792, [0], True);  view_792 = None
    view_793: "f32[8192]" = torch.ops.aten.view.default(sum_154, [8192]);  sum_154 = None
    permute_606: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    view_794: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_197, [1, 128, 2048]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_463: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_794, primals_179);  primals_179 = None
    mul_464: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_463, 2048)
    sum_155: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True)
    mul_465: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_463, mul_106);  mul_463 = None
    sum_156: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [2], True);  mul_465 = None
    mul_466: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, sum_156);  sum_156 = None
    sub_161: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_464, sum_155);  mul_464 = sum_155 = None
    sub_162: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_161, mul_466);  sub_161 = mul_466 = None
    mul_467: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_47, sub_162);  div_47 = sub_162 = None
    mul_468: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_794, mul_106);  mul_106 = None
    sum_157: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 1]);  mul_468 = None
    sum_158: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_794, [0, 1]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_278: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_275, mul_467);  add_275 = mul_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_795: "f32[128, 2048]" = torch.ops.aten.view.default(add_278, [128, 2048])
    mm_199: "f32[128, 2048]" = torch.ops.aten.mm.default(view_795, permute_607);  permute_607 = None
    permute_608: "f32[2048, 128]" = torch.ops.aten.permute.default(view_795, [1, 0])
    mm_200: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_608, view_304);  permute_608 = view_304 = None
    permute_609: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_159: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True);  view_795 = None
    view_796: "f32[2048]" = torch.ops.aten.view.default(sum_159, [2048]);  sum_159 = None
    permute_610: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    view_797: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_199, [1, 128, 2048]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_798: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_797, [1, 128, 16, 128]);  view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_611: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_799: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_611, [16, 128, 128]);  permute_611 = None
    bmm_88: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_612, view_799);  permute_612 = None
    bmm_89: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_799, permute_613);  view_799 = permute_613 = None
    view_800: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_88, [1, 16, 128, 128]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_279: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_30, view_800);  tangents_30 = view_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_801: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_469: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_801, alias_71);  view_801 = None
    sum_160: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [-1], True)
    mul_470: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_71, sum_160);  alias_71 = sum_160 = None
    sub_163: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_38: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_56, sub_163, full_default_25);  slice_56 = sub_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_802: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_38, [16, 128, 128]);  where_38 = None
    bmm_90: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_614, view_802);  permute_614 = None
    bmm_91: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_802, permute_615);  view_802 = permute_615 = None
    view_803: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_90, [1, 16, 128, 128]);  bmm_90 = None
    view_804: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_91, [1, 16, 128, 128]);  bmm_91 = None
    permute_616: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_803, [0, 1, 3, 2]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_280: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_29, permute_616);  tangents_29 = permute_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_617: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_279, [0, 2, 1, 3]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_127: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
    view_805: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_127, [1, 128, 2048]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_618: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_280, [0, 2, 1, 3]);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_128: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_618, memory_format = torch.contiguous_format);  permute_618 = None
    view_806: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_128, [1, 128, 2048]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_619: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_129: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
    view_807: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_129, [1, 128, 2048]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_808: "f32[128, 2048]" = torch.ops.aten.view.default(view_805, [128, 2048]);  view_805 = None
    permute_620: "f32[2048, 128]" = torch.ops.aten.permute.default(view_808, [1, 0])
    mm_201: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_620, view_288);  permute_620 = None
    permute_621: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    mm_202: "f32[128, 2048]" = torch.ops.aten.mm.default(view_808, permute_622);  view_808 = permute_622 = None
    view_809: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_202, [1, 128, 2048]);  mm_202 = None
    permute_623: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_621, [1, 0]);  permute_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_810: "f32[128, 2048]" = torch.ops.aten.view.default(view_806, [128, 2048]);  view_806 = None
    permute_624: "f32[2048, 128]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_203: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_624, view_288);  permute_624 = None
    permute_625: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    mm_204: "f32[128, 2048]" = torch.ops.aten.mm.default(view_810, permute_626);  view_810 = permute_626 = None
    view_811: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_204, [1, 128, 2048]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_281: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_809, view_811);  view_809 = view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_627: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_625, [1, 0]);  permute_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_812: "f32[128, 2048]" = torch.ops.aten.view.default(view_807, [128, 2048]);  view_807 = None
    permute_628: "f32[2048, 128]" = torch.ops.aten.permute.default(view_812, [1, 0])
    mm_205: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_628, view_288);  permute_628 = view_288 = None
    permute_629: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    mm_206: "f32[128, 2048]" = torch.ops.aten.mm.default(view_812, permute_630);  view_812 = permute_630 = None
    view_813: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_206, [1, 128, 2048]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_282: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_281, view_813);  add_281 = view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_631: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_472: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_282, primals_172);  primals_172 = None
    mul_473: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_472, 2048)
    sum_161: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True)
    mul_474: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_472, mul_104);  mul_472 = None
    sum_162: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True);  mul_474 = None
    mul_475: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, sum_162);  sum_162 = None
    sub_165: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_473, sum_161);  mul_473 = sum_161 = None
    sub_166: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_165, mul_475);  sub_165 = mul_475 = None
    mul_476: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_48, sub_166);  div_48 = sub_166 = None
    mul_477: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_282, mul_104);  mul_104 = None
    sum_163: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 1]);  mul_477 = None
    sum_164: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_283: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_278, mul_476);  add_278 = mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_814: "f32[128, 2048]" = torch.ops.aten.view.default(add_283, [128, 2048])
    mm_207: "f32[128, 8192]" = torch.ops.aten.mm.default(view_814, permute_632);  permute_632 = None
    permute_633: "f32[2048, 128]" = torch.ops.aten.permute.default(view_814, [1, 0])
    mm_208: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_633, view_286);  permute_633 = view_286 = None
    permute_634: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    sum_165: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_814, [0], True);  view_814 = None
    view_815: "f32[2048]" = torch.ops.aten.view.default(sum_165, [2048]);  sum_165 = None
    permute_635: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
    view_816: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_207, [1, 128, 8192]);  mm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_478: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_816, mul_100);  mul_100 = None
    mul_479: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_816, add_103);  view_816 = add_103 = None
    alias_72: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    mul_480: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_72, alias_72);  alias_72 = None
    sub_167: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_480);  mul_480 = None
    mul_481: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_478, sub_167);  mul_478 = sub_167 = None
    mul_482: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_481, 0.7978845608028654);  mul_481 = None
    mul_483: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_482, 0.044715)
    pow_36: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_285, 2.0);  view_285 = None
    mul_484: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
    mul_485: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_284: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_482, mul_485);  mul_482 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_486: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_479, 0.5);  mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_285: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_284, mul_486);  add_284 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_817: "f32[128, 8192]" = torch.ops.aten.view.default(add_285, [128, 8192]);  add_285 = None
    mm_209: "f32[128, 2048]" = torch.ops.aten.mm.default(view_817, permute_636);  permute_636 = None
    permute_637: "f32[8192, 128]" = torch.ops.aten.permute.default(view_817, [1, 0])
    mm_210: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_637, view_284);  permute_637 = view_284 = None
    permute_638: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_210, [1, 0]);  mm_210 = None
    sum_166: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_817, [0], True);  view_817 = None
    view_818: "f32[8192]" = torch.ops.aten.view.default(sum_166, [8192]);  sum_166 = None
    permute_639: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    view_819: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_209, [1, 128, 2048]);  mm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_488: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_819, primals_166);  primals_166 = None
    mul_489: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_488, 2048)
    sum_167: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True)
    mul_490: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_488, mul_98);  mul_488 = None
    sum_168: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
    mul_491: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, sum_168);  sum_168 = None
    sub_169: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_489, sum_167);  mul_489 = sum_167 = None
    sub_170: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_169, mul_491);  sub_169 = mul_491 = None
    mul_492: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_49, sub_170);  div_49 = sub_170 = None
    mul_493: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_819, mul_98);  mul_98 = None
    sum_169: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1]);  mul_493 = None
    sum_170: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_819, [0, 1]);  view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_286: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_283, mul_492);  add_283 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_820: "f32[128, 2048]" = torch.ops.aten.view.default(add_286, [128, 2048])
    mm_211: "f32[128, 2048]" = torch.ops.aten.mm.default(view_820, permute_640);  permute_640 = None
    permute_641: "f32[2048, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
    mm_212: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_641, view_282);  permute_641 = view_282 = None
    permute_642: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_212, [1, 0]);  mm_212 = None
    sum_171: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True);  view_820 = None
    view_821: "f32[2048]" = torch.ops.aten.view.default(sum_171, [2048]);  sum_171 = None
    permute_643: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    view_822: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_211, [1, 128, 2048]);  mm_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_823: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_822, [1, 128, 16, 128]);  view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_644: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_823, [0, 2, 1, 3]);  view_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_824: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_644, [16, 128, 128]);  permute_644 = None
    bmm_92: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_645, view_824);  permute_645 = None
    bmm_93: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_824, permute_646);  view_824 = permute_646 = None
    view_825: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_92, [1, 16, 128, 128]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_287: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_28, view_825);  tangents_28 = view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_826: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_494: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_826, alias_73);  view_826 = None
    sum_172: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [-1], True)
    mul_495: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_73, sum_172);  alias_73 = sum_172 = None
    sub_171: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_39: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_52, sub_171, full_default_25);  slice_52 = sub_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_827: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_39, [16, 128, 128]);  where_39 = None
    bmm_94: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_647, view_827);  permute_647 = None
    bmm_95: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_827, permute_648);  view_827 = permute_648 = None
    view_828: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_94, [1, 16, 128, 128]);  bmm_94 = None
    view_829: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_95, [1, 16, 128, 128]);  bmm_95 = None
    permute_649: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_828, [0, 1, 3, 2]);  view_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_288: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_27, permute_649);  tangents_27 = permute_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_650: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_287, [0, 2, 1, 3]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_130: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_830: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_130, [1, 128, 2048]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_651: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_288, [0, 2, 1, 3]);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_131: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_651, memory_format = torch.contiguous_format);  permute_651 = None
    view_831: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_131, [1, 128, 2048]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_652: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_829, [0, 2, 1, 3]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_132: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
    view_832: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_132, [1, 128, 2048]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_833: "f32[128, 2048]" = torch.ops.aten.view.default(view_830, [128, 2048]);  view_830 = None
    permute_653: "f32[2048, 128]" = torch.ops.aten.permute.default(view_833, [1, 0])
    mm_213: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_653, view_266);  permute_653 = None
    permute_654: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    mm_214: "f32[128, 2048]" = torch.ops.aten.mm.default(view_833, permute_655);  view_833 = permute_655 = None
    view_834: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_214, [1, 128, 2048]);  mm_214 = None
    permute_656: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_654, [1, 0]);  permute_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_835: "f32[128, 2048]" = torch.ops.aten.view.default(view_831, [128, 2048]);  view_831 = None
    permute_657: "f32[2048, 128]" = torch.ops.aten.permute.default(view_835, [1, 0])
    mm_215: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_657, view_266);  permute_657 = None
    permute_658: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    mm_216: "f32[128, 2048]" = torch.ops.aten.mm.default(view_835, permute_659);  view_835 = permute_659 = None
    view_836: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_216, [1, 128, 2048]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_289: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_834, view_836);  view_834 = view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_660: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_658, [1, 0]);  permute_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_837: "f32[128, 2048]" = torch.ops.aten.view.default(view_832, [128, 2048]);  view_832 = None
    permute_661: "f32[2048, 128]" = torch.ops.aten.permute.default(view_837, [1, 0])
    mm_217: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_661, view_266);  permute_661 = view_266 = None
    permute_662: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    mm_218: "f32[128, 2048]" = torch.ops.aten.mm.default(view_837, permute_663);  view_837 = permute_663 = None
    view_838: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_218, [1, 128, 2048]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_290: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_289, view_838);  add_289 = view_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_664: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_662, [1, 0]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_497: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_290, primals_159);  primals_159 = None
    mul_498: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_497, 2048)
    sum_173: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True)
    mul_499: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_497, mul_96);  mul_497 = None
    sum_174: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True);  mul_499 = None
    mul_500: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, sum_174);  sum_174 = None
    sub_173: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_498, sum_173);  mul_498 = sum_173 = None
    sub_174: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_173, mul_500);  sub_173 = mul_500 = None
    mul_501: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_50, sub_174);  div_50 = sub_174 = None
    mul_502: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_290, mul_96);  mul_96 = None
    sum_175: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 1]);  mul_502 = None
    sum_176: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_291: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_286, mul_501);  add_286 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_839: "f32[128, 2048]" = torch.ops.aten.view.default(add_291, [128, 2048])
    mm_219: "f32[128, 8192]" = torch.ops.aten.mm.default(view_839, permute_665);  permute_665 = None
    permute_666: "f32[2048, 128]" = torch.ops.aten.permute.default(view_839, [1, 0])
    mm_220: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_666, view_264);  permute_666 = view_264 = None
    permute_667: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_220, [1, 0]);  mm_220 = None
    sum_177: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_839, [0], True);  view_839 = None
    view_840: "f32[2048]" = torch.ops.aten.view.default(sum_177, [2048]);  sum_177 = None
    permute_668: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_841: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_219, [1, 128, 8192]);  mm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_503: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_841, mul_92);  mul_92 = None
    mul_504: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_841, add_95);  view_841 = add_95 = None
    alias_74: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_505: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_74, alias_74);  alias_74 = None
    sub_175: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_505);  mul_505 = None
    mul_506: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_503, sub_175);  mul_503 = sub_175 = None
    mul_507: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_506, 0.7978845608028654);  mul_506 = None
    mul_508: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_507, 0.044715)
    pow_37: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
    mul_509: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
    mul_510: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_292: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_507, mul_510);  mul_507 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_511: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_504, 0.5);  mul_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_293: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_292, mul_511);  add_292 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_842: "f32[128, 8192]" = torch.ops.aten.view.default(add_293, [128, 8192]);  add_293 = None
    mm_221: "f32[128, 2048]" = torch.ops.aten.mm.default(view_842, permute_669);  permute_669 = None
    permute_670: "f32[8192, 128]" = torch.ops.aten.permute.default(view_842, [1, 0])
    mm_222: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_670, view_262);  permute_670 = view_262 = None
    permute_671: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_222, [1, 0]);  mm_222 = None
    sum_178: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
    view_843: "f32[8192]" = torch.ops.aten.view.default(sum_178, [8192]);  sum_178 = None
    permute_672: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    view_844: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_221, [1, 128, 2048]);  mm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_513: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_844, primals_153);  primals_153 = None
    mul_514: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_513, 2048)
    sum_179: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True)
    mul_515: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_513, mul_90);  mul_513 = None
    sum_180: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True);  mul_515 = None
    mul_516: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, sum_180);  sum_180 = None
    sub_177: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_514, sum_179);  mul_514 = sum_179 = None
    sub_178: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_177, mul_516);  sub_177 = mul_516 = None
    mul_517: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_51, sub_178);  div_51 = sub_178 = None
    mul_518: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_844, mul_90);  mul_90 = None
    sum_181: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 1]);  mul_518 = None
    sum_182: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_844, [0, 1]);  view_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_294: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_291, mul_517);  add_291 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_845: "f32[128, 2048]" = torch.ops.aten.view.default(add_294, [128, 2048])
    mm_223: "f32[128, 2048]" = torch.ops.aten.mm.default(view_845, permute_673);  permute_673 = None
    permute_674: "f32[2048, 128]" = torch.ops.aten.permute.default(view_845, [1, 0])
    mm_224: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_674, view_260);  permute_674 = view_260 = None
    permute_675: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    sum_183: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_845, [0], True);  view_845 = None
    view_846: "f32[2048]" = torch.ops.aten.view.default(sum_183, [2048]);  sum_183 = None
    permute_676: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    view_847: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_223, [1, 128, 2048]);  mm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_848: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_847, [1, 128, 16, 128]);  view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_677: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_849: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_677, [16, 128, 128]);  permute_677 = None
    bmm_96: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_678, view_849);  permute_678 = None
    bmm_97: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_849, permute_679);  view_849 = permute_679 = None
    view_850: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_96, [1, 16, 128, 128]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_295: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_26, view_850);  tangents_26 = view_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_851: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_519: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_851, alias_75);  view_851 = None
    sum_184: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [-1], True)
    mul_520: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_75, sum_184);  alias_75 = sum_184 = None
    sub_179: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_40: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, sub_179, full_default_25);  slice_48 = sub_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_852: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_40, [16, 128, 128]);  where_40 = None
    bmm_98: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_680, view_852);  permute_680 = None
    bmm_99: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_852, permute_681);  view_852 = permute_681 = None
    view_853: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_98, [1, 16, 128, 128]);  bmm_98 = None
    view_854: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_99, [1, 16, 128, 128]);  bmm_99 = None
    permute_682: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_853, [0, 1, 3, 2]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_296: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_25, permute_682);  tangents_25 = permute_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_683: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_295, [0, 2, 1, 3]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_133: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
    view_855: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_133, [1, 128, 2048]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_684: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_296, [0, 2, 1, 3]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_134: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
    view_856: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_134, [1, 128, 2048]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_685: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_135: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_857: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_135, [1, 128, 2048]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_858: "f32[128, 2048]" = torch.ops.aten.view.default(view_855, [128, 2048]);  view_855 = None
    permute_686: "f32[2048, 128]" = torch.ops.aten.permute.default(view_858, [1, 0])
    mm_225: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_686, view_244);  permute_686 = None
    permute_687: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    mm_226: "f32[128, 2048]" = torch.ops.aten.mm.default(view_858, permute_688);  view_858 = permute_688 = None
    view_859: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_226, [1, 128, 2048]);  mm_226 = None
    permute_689: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_860: "f32[128, 2048]" = torch.ops.aten.view.default(view_856, [128, 2048]);  view_856 = None
    permute_690: "f32[2048, 128]" = torch.ops.aten.permute.default(view_860, [1, 0])
    mm_227: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_690, view_244);  permute_690 = None
    permute_691: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    mm_228: "f32[128, 2048]" = torch.ops.aten.mm.default(view_860, permute_692);  view_860 = permute_692 = None
    view_861: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_228, [1, 128, 2048]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_297: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_859, view_861);  view_859 = view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_693: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_691, [1, 0]);  permute_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_862: "f32[128, 2048]" = torch.ops.aten.view.default(view_857, [128, 2048]);  view_857 = None
    permute_694: "f32[2048, 128]" = torch.ops.aten.permute.default(view_862, [1, 0])
    mm_229: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_694, view_244);  permute_694 = view_244 = None
    permute_695: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    mm_230: "f32[128, 2048]" = torch.ops.aten.mm.default(view_862, permute_696);  view_862 = permute_696 = None
    view_863: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_230, [1, 128, 2048]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_298: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_297, view_863);  add_297 = view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_697: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_695, [1, 0]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_522: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_298, primals_146);  primals_146 = None
    mul_523: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_522, 2048)
    sum_185: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
    mul_524: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_522, mul_88);  mul_522 = None
    sum_186: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
    mul_525: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, sum_186);  sum_186 = None
    sub_181: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_523, sum_185);  mul_523 = sum_185 = None
    sub_182: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_181, mul_525);  sub_181 = mul_525 = None
    mul_526: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_52, sub_182);  div_52 = sub_182 = None
    mul_527: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_298, mul_88);  mul_88 = None
    sum_187: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
    sum_188: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_298, [0, 1]);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_299: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_294, mul_526);  add_294 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_864: "f32[128, 2048]" = torch.ops.aten.view.default(add_299, [128, 2048])
    mm_231: "f32[128, 8192]" = torch.ops.aten.mm.default(view_864, permute_698);  permute_698 = None
    permute_699: "f32[2048, 128]" = torch.ops.aten.permute.default(view_864, [1, 0])
    mm_232: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_699, view_242);  permute_699 = view_242 = None
    permute_700: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    sum_189: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
    view_865: "f32[2048]" = torch.ops.aten.view.default(sum_189, [2048]);  sum_189 = None
    permute_701: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_700, [1, 0]);  permute_700 = None
    view_866: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_231, [1, 128, 8192]);  mm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_528: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_866, mul_84);  mul_84 = None
    mul_529: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_866, add_87);  view_866 = add_87 = None
    alias_76: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_530: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_76, alias_76);  alias_76 = None
    sub_183: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_530);  mul_530 = None
    mul_531: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_528, sub_183);  mul_528 = sub_183 = None
    mul_532: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_531, 0.7978845608028654);  mul_531 = None
    mul_533: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_532, 0.044715)
    pow_38: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
    mul_534: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
    mul_535: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_300: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_532, mul_535);  mul_532 = mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_536: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_529, 0.5);  mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_301: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_300, mul_536);  add_300 = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_867: "f32[128, 8192]" = torch.ops.aten.view.default(add_301, [128, 8192]);  add_301 = None
    mm_233: "f32[128, 2048]" = torch.ops.aten.mm.default(view_867, permute_702);  permute_702 = None
    permute_703: "f32[8192, 128]" = torch.ops.aten.permute.default(view_867, [1, 0])
    mm_234: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_703, view_240);  permute_703 = view_240 = None
    permute_704: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_234, [1, 0]);  mm_234 = None
    sum_190: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_867, [0], True);  view_867 = None
    view_868: "f32[8192]" = torch.ops.aten.view.default(sum_190, [8192]);  sum_190 = None
    permute_705: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    view_869: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_233, [1, 128, 2048]);  mm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_538: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_869, primals_140);  primals_140 = None
    mul_539: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_538, 2048)
    sum_191: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True)
    mul_540: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_538, mul_82);  mul_538 = None
    sum_192: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [2], True);  mul_540 = None
    mul_541: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, sum_192);  sum_192 = None
    sub_185: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_539, sum_191);  mul_539 = sum_191 = None
    sub_186: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_185, mul_541);  sub_185 = mul_541 = None
    mul_542: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_53, sub_186);  div_53 = sub_186 = None
    mul_543: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_869, mul_82);  mul_82 = None
    sum_193: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1]);  mul_543 = None
    sum_194: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_869, [0, 1]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_302: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_299, mul_542);  add_299 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_870: "f32[128, 2048]" = torch.ops.aten.view.default(add_302, [128, 2048])
    mm_235: "f32[128, 2048]" = torch.ops.aten.mm.default(view_870, permute_706);  permute_706 = None
    permute_707: "f32[2048, 128]" = torch.ops.aten.permute.default(view_870, [1, 0])
    mm_236: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_707, view_238);  permute_707 = view_238 = None
    permute_708: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_236, [1, 0]);  mm_236 = None
    sum_195: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
    view_871: "f32[2048]" = torch.ops.aten.view.default(sum_195, [2048]);  sum_195 = None
    permute_709: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    view_872: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_235, [1, 128, 2048]);  mm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_873: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_872, [1, 128, 16, 128]);  view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_710: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_874: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_710, [16, 128, 128]);  permute_710 = None
    bmm_100: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_711, view_874);  permute_711 = None
    bmm_101: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_874, permute_712);  view_874 = permute_712 = None
    view_875: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_100, [1, 16, 128, 128]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_303: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_24, view_875);  tangents_24 = view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_876: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_544: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_876, alias_77);  view_876 = None
    sum_196: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [-1], True)
    mul_545: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_77, sum_196);  alias_77 = sum_196 = None
    sub_187: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_41: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, sub_187, full_default_25);  slice_44 = sub_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_877: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_41, [16, 128, 128]);  where_41 = None
    bmm_102: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_713, view_877);  permute_713 = None
    bmm_103: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_877, permute_714);  view_877 = permute_714 = None
    view_878: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_102, [1, 16, 128, 128]);  bmm_102 = None
    view_879: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_103, [1, 16, 128, 128]);  bmm_103 = None
    permute_715: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_878, [0, 1, 3, 2]);  view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_304: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_23, permute_715);  tangents_23 = permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_716: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_303, [0, 2, 1, 3]);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_136: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_880: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_136, [1, 128, 2048]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_717: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_304, [0, 2, 1, 3]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_137: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_717, memory_format = torch.contiguous_format);  permute_717 = None
    view_881: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_137, [1, 128, 2048]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_718: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_879, [0, 2, 1, 3]);  view_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_138: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
    view_882: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_138, [1, 128, 2048]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_883: "f32[128, 2048]" = torch.ops.aten.view.default(view_880, [128, 2048]);  view_880 = None
    permute_719: "f32[2048, 128]" = torch.ops.aten.permute.default(view_883, [1, 0])
    mm_237: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_719, view_222);  permute_719 = None
    permute_720: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    mm_238: "f32[128, 2048]" = torch.ops.aten.mm.default(view_883, permute_721);  view_883 = permute_721 = None
    view_884: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_238, [1, 128, 2048]);  mm_238 = None
    permute_722: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_720, [1, 0]);  permute_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_885: "f32[128, 2048]" = torch.ops.aten.view.default(view_881, [128, 2048]);  view_881 = None
    permute_723: "f32[2048, 128]" = torch.ops.aten.permute.default(view_885, [1, 0])
    mm_239: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_723, view_222);  permute_723 = None
    permute_724: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    mm_240: "f32[128, 2048]" = torch.ops.aten.mm.default(view_885, permute_725);  view_885 = permute_725 = None
    view_886: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_240, [1, 128, 2048]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_305: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_884, view_886);  view_884 = view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_726: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_724, [1, 0]);  permute_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_887: "f32[128, 2048]" = torch.ops.aten.view.default(view_882, [128, 2048]);  view_882 = None
    permute_727: "f32[2048, 128]" = torch.ops.aten.permute.default(view_887, [1, 0])
    mm_241: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_727, view_222);  permute_727 = view_222 = None
    permute_728: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    mm_242: "f32[128, 2048]" = torch.ops.aten.mm.default(view_887, permute_729);  view_887 = permute_729 = None
    view_888: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_242, [1, 128, 2048]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_306: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_305, view_888);  add_305 = view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_730: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_547: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_306, primals_133);  primals_133 = None
    mul_548: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_547, 2048)
    sum_197: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True)
    mul_549: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_547, mul_80);  mul_547 = None
    sum_198: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True);  mul_549 = None
    mul_550: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, sum_198);  sum_198 = None
    sub_189: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_548, sum_197);  mul_548 = sum_197 = None
    sub_190: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_189, mul_550);  sub_189 = mul_550 = None
    mul_551: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_54, sub_190);  div_54 = sub_190 = None
    mul_552: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_306, mul_80);  mul_80 = None
    sum_199: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 1]);  mul_552 = None
    sum_200: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_306, [0, 1]);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_307: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_302, mul_551);  add_302 = mul_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_889: "f32[128, 2048]" = torch.ops.aten.view.default(add_307, [128, 2048])
    mm_243: "f32[128, 8192]" = torch.ops.aten.mm.default(view_889, permute_731);  permute_731 = None
    permute_732: "f32[2048, 128]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_244: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_732, view_220);  permute_732 = view_220 = None
    permute_733: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    sum_201: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[2048]" = torch.ops.aten.view.default(sum_201, [2048]);  sum_201 = None
    permute_734: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    view_891: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_243, [1, 128, 8192]);  mm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_553: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_891, mul_76);  mul_76 = None
    mul_554: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_891, add_79);  view_891 = add_79 = None
    alias_78: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_555: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_78, alias_78);  alias_78 = None
    sub_191: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_555);  mul_555 = None
    mul_556: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_553, sub_191);  mul_553 = sub_191 = None
    mul_557: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_556, 0.7978845608028654);  mul_556 = None
    mul_558: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_557, 0.044715)
    pow_39: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
    mul_559: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
    mul_560: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_558, mul_559);  mul_558 = mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_308: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_557, mul_560);  mul_557 = mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_561: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_554, 0.5);  mul_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_309: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_308, mul_561);  add_308 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_892: "f32[128, 8192]" = torch.ops.aten.view.default(add_309, [128, 8192]);  add_309 = None
    mm_245: "f32[128, 2048]" = torch.ops.aten.mm.default(view_892, permute_735);  permute_735 = None
    permute_736: "f32[8192, 128]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_246: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_736, view_218);  permute_736 = view_218 = None
    permute_737: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_246, [1, 0]);  mm_246 = None
    sum_202: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[8192]" = torch.ops.aten.view.default(sum_202, [8192]);  sum_202 = None
    permute_738: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    view_894: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_245, [1, 128, 2048]);  mm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_563: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_894, primals_127);  primals_127 = None
    mul_564: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_563, 2048)
    sum_203: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True)
    mul_565: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_563, mul_74);  mul_563 = None
    sum_204: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True);  mul_565 = None
    mul_566: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, sum_204);  sum_204 = None
    sub_193: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_564, sum_203);  mul_564 = sum_203 = None
    sub_194: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_193, mul_566);  sub_193 = mul_566 = None
    mul_567: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_55, sub_194);  div_55 = sub_194 = None
    mul_568: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_894, mul_74);  mul_74 = None
    sum_205: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1]);  mul_568 = None
    sum_206: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_894, [0, 1]);  view_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_310: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_307, mul_567);  add_307 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_895: "f32[128, 2048]" = torch.ops.aten.view.default(add_310, [128, 2048])
    mm_247: "f32[128, 2048]" = torch.ops.aten.mm.default(view_895, permute_739);  permute_739 = None
    permute_740: "f32[2048, 128]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_248: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_740, view_216);  permute_740 = view_216 = None
    permute_741: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_248, [1, 0]);  mm_248 = None
    sum_207: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[2048]" = torch.ops.aten.view.default(sum_207, [2048]);  sum_207 = None
    permute_742: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_741, [1, 0]);  permute_741 = None
    view_897: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_247, [1, 128, 2048]);  mm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_898: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_897, [1, 128, 16, 128]);  view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_743: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_898, [0, 2, 1, 3]);  view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_899: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_743, [16, 128, 128]);  permute_743 = None
    bmm_104: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_744, view_899);  permute_744 = None
    bmm_105: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_899, permute_745);  view_899 = permute_745 = None
    view_900: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_104, [1, 16, 128, 128]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_311: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_22, view_900);  tangents_22 = view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_901: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_569: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_901, alias_79);  view_901 = None
    sum_208: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [-1], True)
    mul_570: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_79, sum_208);  alias_79 = sum_208 = None
    sub_195: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_42: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, sub_195, full_default_25);  slice_40 = sub_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_902: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_42, [16, 128, 128]);  where_42 = None
    bmm_106: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_746, view_902);  permute_746 = None
    bmm_107: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_902, permute_747);  view_902 = permute_747 = None
    view_903: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_106, [1, 16, 128, 128]);  bmm_106 = None
    view_904: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_107, [1, 16, 128, 128]);  bmm_107 = None
    permute_748: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_903, [0, 1, 3, 2]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_312: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_21, permute_748);  tangents_21 = permute_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_749: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_311, [0, 2, 1, 3]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_139: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
    view_905: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_139, [1, 128, 2048]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_750: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_312, [0, 2, 1, 3]);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_140: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
    view_906: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_140, [1, 128, 2048]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_751: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_904, [0, 2, 1, 3]);  view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_141: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
    view_907: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_141, [1, 128, 2048]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_908: "f32[128, 2048]" = torch.ops.aten.view.default(view_905, [128, 2048]);  view_905 = None
    permute_752: "f32[2048, 128]" = torch.ops.aten.permute.default(view_908, [1, 0])
    mm_249: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_752, view_200);  permute_752 = None
    permute_753: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    mm_250: "f32[128, 2048]" = torch.ops.aten.mm.default(view_908, permute_754);  view_908 = permute_754 = None
    view_909: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_250, [1, 128, 2048]);  mm_250 = None
    permute_755: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_910: "f32[128, 2048]" = torch.ops.aten.view.default(view_906, [128, 2048]);  view_906 = None
    permute_756: "f32[2048, 128]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_251: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_756, view_200);  permute_756 = None
    permute_757: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    mm_252: "f32[128, 2048]" = torch.ops.aten.mm.default(view_910, permute_758);  view_910 = permute_758 = None
    view_911: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_252, [1, 128, 2048]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_313: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_909, view_911);  view_909 = view_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_759: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_912: "f32[128, 2048]" = torch.ops.aten.view.default(view_907, [128, 2048]);  view_907 = None
    permute_760: "f32[2048, 128]" = torch.ops.aten.permute.default(view_912, [1, 0])
    mm_253: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_760, view_200);  permute_760 = view_200 = None
    permute_761: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    mm_254: "f32[128, 2048]" = torch.ops.aten.mm.default(view_912, permute_762);  view_912 = permute_762 = None
    view_913: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_254, [1, 128, 2048]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_314: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_313, view_913);  add_313 = view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_763: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_761, [1, 0]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_572: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_314, primals_120);  primals_120 = None
    mul_573: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_572, 2048)
    sum_209: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True)
    mul_574: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_572, mul_72);  mul_572 = None
    sum_210: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True);  mul_574 = None
    mul_575: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, sum_210);  sum_210 = None
    sub_197: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_573, sum_209);  mul_573 = sum_209 = None
    sub_198: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_197, mul_575);  sub_197 = mul_575 = None
    mul_576: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_56, sub_198);  div_56 = sub_198 = None
    mul_577: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_314, mul_72);  mul_72 = None
    sum_211: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 1]);  mul_577 = None
    sum_212: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_315: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_310, mul_576);  add_310 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_914: "f32[128, 2048]" = torch.ops.aten.view.default(add_315, [128, 2048])
    mm_255: "f32[128, 8192]" = torch.ops.aten.mm.default(view_914, permute_764);  permute_764 = None
    permute_765: "f32[2048, 128]" = torch.ops.aten.permute.default(view_914, [1, 0])
    mm_256: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_765, view_198);  permute_765 = view_198 = None
    permute_766: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    sum_213: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_914, [0], True);  view_914 = None
    view_915: "f32[2048]" = torch.ops.aten.view.default(sum_213, [2048]);  sum_213 = None
    permute_767: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_766, [1, 0]);  permute_766 = None
    view_916: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_255, [1, 128, 8192]);  mm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_578: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_916, mul_68);  mul_68 = None
    mul_579: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_916, add_71);  view_916 = add_71 = None
    alias_80: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_580: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_80, alias_80);  alias_80 = None
    sub_199: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_580);  mul_580 = None
    mul_581: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_578, sub_199);  mul_578 = sub_199 = None
    mul_582: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_581, 0.7978845608028654);  mul_581 = None
    mul_583: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_582, 0.044715)
    pow_40: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_584: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
    mul_585: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_316: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_582, mul_585);  mul_582 = mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_586: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_579, 0.5);  mul_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_317: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_316, mul_586);  add_316 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_917: "f32[128, 8192]" = torch.ops.aten.view.default(add_317, [128, 8192]);  add_317 = None
    mm_257: "f32[128, 2048]" = torch.ops.aten.mm.default(view_917, permute_768);  permute_768 = None
    permute_769: "f32[8192, 128]" = torch.ops.aten.permute.default(view_917, [1, 0])
    mm_258: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_769, view_196);  permute_769 = view_196 = None
    permute_770: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_258, [1, 0]);  mm_258 = None
    sum_214: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_917, [0], True);  view_917 = None
    view_918: "f32[8192]" = torch.ops.aten.view.default(sum_214, [8192]);  sum_214 = None
    permute_771: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_770, [1, 0]);  permute_770 = None
    view_919: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_257, [1, 128, 2048]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_588: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_919, primals_114);  primals_114 = None
    mul_589: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_588, 2048)
    sum_215: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True)
    mul_590: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_588, mul_66);  mul_588 = None
    sum_216: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True);  mul_590 = None
    mul_591: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, sum_216);  sum_216 = None
    sub_201: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_589, sum_215);  mul_589 = sum_215 = None
    sub_202: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_201, mul_591);  sub_201 = mul_591 = None
    mul_592: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_57, sub_202);  div_57 = sub_202 = None
    mul_593: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_919, mul_66);  mul_66 = None
    sum_217: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 1]);  mul_593 = None
    sum_218: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_919, [0, 1]);  view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_318: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_315, mul_592);  add_315 = mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_920: "f32[128, 2048]" = torch.ops.aten.view.default(add_318, [128, 2048])
    mm_259: "f32[128, 2048]" = torch.ops.aten.mm.default(view_920, permute_772);  permute_772 = None
    permute_773: "f32[2048, 128]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_260: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_773, view_194);  permute_773 = view_194 = None
    permute_774: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_260, [1, 0]);  mm_260 = None
    sum_219: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
    view_921: "f32[2048]" = torch.ops.aten.view.default(sum_219, [2048]);  sum_219 = None
    permute_775: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
    view_922: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_259, [1, 128, 2048]);  mm_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_923: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_922, [1, 128, 16, 128]);  view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_776: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_924: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_776, [16, 128, 128]);  permute_776 = None
    bmm_108: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_777, view_924);  permute_777 = None
    bmm_109: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_924, permute_778);  view_924 = permute_778 = None
    view_925: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_108, [1, 16, 128, 128]);  bmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_319: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_20, view_925);  tangents_20 = view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_926: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_594: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_926, alias_81);  view_926 = None
    sum_220: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [-1], True)
    mul_595: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_81, sum_220);  alias_81 = sum_220 = None
    sub_203: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_594, mul_595);  mul_594 = mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_43: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, sub_203, full_default_25);  slice_36 = sub_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_927: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_43, [16, 128, 128]);  where_43 = None
    bmm_110: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_779, view_927);  permute_779 = None
    bmm_111: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_927, permute_780);  view_927 = permute_780 = None
    view_928: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_110, [1, 16, 128, 128]);  bmm_110 = None
    view_929: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_111, [1, 16, 128, 128]);  bmm_111 = None
    permute_781: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_928, [0, 1, 3, 2]);  view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_320: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_19, permute_781);  tangents_19 = permute_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_782: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_319, [0, 2, 1, 3]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_142: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
    view_930: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_142, [1, 128, 2048]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_783: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_320, [0, 2, 1, 3]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_143: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
    view_931: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_143, [1, 128, 2048]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_784: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_929, [0, 2, 1, 3]);  view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_144: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
    view_932: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_144, [1, 128, 2048]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_933: "f32[128, 2048]" = torch.ops.aten.view.default(view_930, [128, 2048]);  view_930 = None
    permute_785: "f32[2048, 128]" = torch.ops.aten.permute.default(view_933, [1, 0])
    mm_261: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_785, view_178);  permute_785 = None
    permute_786: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    mm_262: "f32[128, 2048]" = torch.ops.aten.mm.default(view_933, permute_787);  view_933 = permute_787 = None
    view_934: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_262, [1, 128, 2048]);  mm_262 = None
    permute_788: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_786, [1, 0]);  permute_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_935: "f32[128, 2048]" = torch.ops.aten.view.default(view_931, [128, 2048]);  view_931 = None
    permute_789: "f32[2048, 128]" = torch.ops.aten.permute.default(view_935, [1, 0])
    mm_263: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_789, view_178);  permute_789 = None
    permute_790: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    mm_264: "f32[128, 2048]" = torch.ops.aten.mm.default(view_935, permute_791);  view_935 = permute_791 = None
    view_936: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_264, [1, 128, 2048]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_321: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_934, view_936);  view_934 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_792: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_790, [1, 0]);  permute_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_937: "f32[128, 2048]" = torch.ops.aten.view.default(view_932, [128, 2048]);  view_932 = None
    permute_793: "f32[2048, 128]" = torch.ops.aten.permute.default(view_937, [1, 0])
    mm_265: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_793, view_178);  permute_793 = view_178 = None
    permute_794: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    mm_266: "f32[128, 2048]" = torch.ops.aten.mm.default(view_937, permute_795);  view_937 = permute_795 = None
    view_938: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_266, [1, 128, 2048]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_322: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_321, view_938);  add_321 = view_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_796: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_794, [1, 0]);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_597: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_322, primals_107);  primals_107 = None
    mul_598: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_597, 2048)
    sum_221: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True)
    mul_599: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_597, mul_64);  mul_597 = None
    sum_222: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True);  mul_599 = None
    mul_600: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, sum_222);  sum_222 = None
    sub_205: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_598, sum_221);  mul_598 = sum_221 = None
    sub_206: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_205, mul_600);  sub_205 = mul_600 = None
    mul_601: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_58, sub_206);  div_58 = sub_206 = None
    mul_602: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_322, mul_64);  mul_64 = None
    sum_223: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 1]);  mul_602 = None
    sum_224: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_322, [0, 1]);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_323: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_318, mul_601);  add_318 = mul_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_939: "f32[128, 2048]" = torch.ops.aten.view.default(add_323, [128, 2048])
    mm_267: "f32[128, 8192]" = torch.ops.aten.mm.default(view_939, permute_797);  permute_797 = None
    permute_798: "f32[2048, 128]" = torch.ops.aten.permute.default(view_939, [1, 0])
    mm_268: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_798, view_176);  permute_798 = view_176 = None
    permute_799: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    sum_225: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
    view_940: "f32[2048]" = torch.ops.aten.view.default(sum_225, [2048]);  sum_225 = None
    permute_800: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
    view_941: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_267, [1, 128, 8192]);  mm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_603: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_941, mul_60);  mul_60 = None
    mul_604: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_941, add_63);  view_941 = add_63 = None
    alias_82: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_605: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_82, alias_82);  alias_82 = None
    sub_207: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_605);  mul_605 = None
    mul_606: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_603, sub_207);  mul_603 = sub_207 = None
    mul_607: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_606, 0.7978845608028654);  mul_606 = None
    mul_608: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_607, 0.044715)
    pow_41: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
    mul_609: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
    mul_610: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_324: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_607, mul_610);  mul_607 = mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_611: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_604, 0.5);  mul_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_325: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_324, mul_611);  add_324 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_942: "f32[128, 8192]" = torch.ops.aten.view.default(add_325, [128, 8192]);  add_325 = None
    mm_269: "f32[128, 2048]" = torch.ops.aten.mm.default(view_942, permute_801);  permute_801 = None
    permute_802: "f32[8192, 128]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_270: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_802, view_174);  permute_802 = view_174 = None
    permute_803: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_270, [1, 0]);  mm_270 = None
    sum_226: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[8192]" = torch.ops.aten.view.default(sum_226, [8192]);  sum_226 = None
    permute_804: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_803, [1, 0]);  permute_803 = None
    view_944: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_269, [1, 128, 2048]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_613: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_944, primals_101);  primals_101 = None
    mul_614: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_613, 2048)
    sum_227: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True)
    mul_615: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_613, mul_58);  mul_613 = None
    sum_228: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True);  mul_615 = None
    mul_616: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, sum_228);  sum_228 = None
    sub_209: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_614, sum_227);  mul_614 = sum_227 = None
    sub_210: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_209, mul_616);  sub_209 = mul_616 = None
    mul_617: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_59, sub_210);  div_59 = sub_210 = None
    mul_618: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_944, mul_58);  mul_58 = None
    sum_229: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 1]);  mul_618 = None
    sum_230: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_944, [0, 1]);  view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_326: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_323, mul_617);  add_323 = mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_945: "f32[128, 2048]" = torch.ops.aten.view.default(add_326, [128, 2048])
    mm_271: "f32[128, 2048]" = torch.ops.aten.mm.default(view_945, permute_805);  permute_805 = None
    permute_806: "f32[2048, 128]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_272: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_806, view_172);  permute_806 = view_172 = None
    permute_807: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_272, [1, 0]);  mm_272 = None
    sum_231: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[2048]" = torch.ops.aten.view.default(sum_231, [2048]);  sum_231 = None
    permute_808: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_807, [1, 0]);  permute_807 = None
    view_947: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_271, [1, 128, 2048]);  mm_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_948: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_947, [1, 128, 16, 128]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_809: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_949: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_809, [16, 128, 128]);  permute_809 = None
    bmm_112: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_810, view_949);  permute_810 = None
    bmm_113: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_949, permute_811);  view_949 = permute_811 = None
    view_950: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_112, [1, 16, 128, 128]);  bmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_327: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_18, view_950);  tangents_18 = view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_951: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_619: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_951, alias_83);  view_951 = None
    sum_232: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [-1], True)
    mul_620: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_83, sum_232);  alias_83 = sum_232 = None
    sub_211: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_44: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, sub_211, full_default_25);  slice_32 = sub_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_952: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_44, [16, 128, 128]);  where_44 = None
    bmm_114: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_812, view_952);  permute_812 = None
    bmm_115: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_952, permute_813);  view_952 = permute_813 = None
    view_953: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_114, [1, 16, 128, 128]);  bmm_114 = None
    view_954: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_115, [1, 16, 128, 128]);  bmm_115 = None
    permute_814: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_953, [0, 1, 3, 2]);  view_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_328: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_17, permute_814);  tangents_17 = permute_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_815: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_327, [0, 2, 1, 3]);  add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_145: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
    view_955: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_145, [1, 128, 2048]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_816: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_328, [0, 2, 1, 3]);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_146: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
    view_956: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_146, [1, 128, 2048]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_817: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_954, [0, 2, 1, 3]);  view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_147: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_817, memory_format = torch.contiguous_format);  permute_817 = None
    view_957: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_147, [1, 128, 2048]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_958: "f32[128, 2048]" = torch.ops.aten.view.default(view_955, [128, 2048]);  view_955 = None
    permute_818: "f32[2048, 128]" = torch.ops.aten.permute.default(view_958, [1, 0])
    mm_273: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_818, view_156);  permute_818 = None
    permute_819: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    mm_274: "f32[128, 2048]" = torch.ops.aten.mm.default(view_958, permute_820);  view_958 = permute_820 = None
    view_959: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_274, [1, 128, 2048]);  mm_274 = None
    permute_821: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_960: "f32[128, 2048]" = torch.ops.aten.view.default(view_956, [128, 2048]);  view_956 = None
    permute_822: "f32[2048, 128]" = torch.ops.aten.permute.default(view_960, [1, 0])
    mm_275: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_822, view_156);  permute_822 = None
    permute_823: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    mm_276: "f32[128, 2048]" = torch.ops.aten.mm.default(view_960, permute_824);  view_960 = permute_824 = None
    view_961: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_276, [1, 128, 2048]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_329: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_959, view_961);  view_959 = view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_825: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_823, [1, 0]);  permute_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_962: "f32[128, 2048]" = torch.ops.aten.view.default(view_957, [128, 2048]);  view_957 = None
    permute_826: "f32[2048, 128]" = torch.ops.aten.permute.default(view_962, [1, 0])
    mm_277: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_826, view_156);  permute_826 = view_156 = None
    permute_827: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    mm_278: "f32[128, 2048]" = torch.ops.aten.mm.default(view_962, permute_828);  view_962 = permute_828 = None
    view_963: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_278, [1, 128, 2048]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_330: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_329, view_963);  add_329 = view_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_829: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_622: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_330, primals_94);  primals_94 = None
    mul_623: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_622, 2048)
    sum_233: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True)
    mul_624: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_622, mul_56);  mul_622 = None
    sum_234: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True);  mul_624 = None
    mul_625: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, sum_234);  sum_234 = None
    sub_213: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_623, sum_233);  mul_623 = sum_233 = None
    sub_214: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_213, mul_625);  sub_213 = mul_625 = None
    mul_626: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_60, sub_214);  div_60 = sub_214 = None
    mul_627: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_330, mul_56);  mul_56 = None
    sum_235: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 1]);  mul_627 = None
    sum_236: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_331: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_326, mul_626);  add_326 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_964: "f32[128, 2048]" = torch.ops.aten.view.default(add_331, [128, 2048])
    mm_279: "f32[128, 8192]" = torch.ops.aten.mm.default(view_964, permute_830);  permute_830 = None
    permute_831: "f32[2048, 128]" = torch.ops.aten.permute.default(view_964, [1, 0])
    mm_280: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_831, view_154);  permute_831 = view_154 = None
    permute_832: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_280, [1, 0]);  mm_280 = None
    sum_237: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
    view_965: "f32[2048]" = torch.ops.aten.view.default(sum_237, [2048]);  sum_237 = None
    permute_833: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_832, [1, 0]);  permute_832 = None
    view_966: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_279, [1, 128, 8192]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_628: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_966, mul_52);  mul_52 = None
    mul_629: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_966, add_55);  view_966 = add_55 = None
    alias_84: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_630: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_84, alias_84);  alias_84 = None
    sub_215: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_630);  mul_630 = None
    mul_631: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_628, sub_215);  mul_628 = sub_215 = None
    mul_632: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_631, 0.7978845608028654);  mul_631 = None
    mul_633: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_632, 0.044715)
    pow_42: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
    mul_634: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
    mul_635: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_332: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_632, mul_635);  mul_632 = mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_636: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_629, 0.5);  mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_333: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_332, mul_636);  add_332 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_967: "f32[128, 8192]" = torch.ops.aten.view.default(add_333, [128, 8192]);  add_333 = None
    mm_281: "f32[128, 2048]" = torch.ops.aten.mm.default(view_967, permute_834);  permute_834 = None
    permute_835: "f32[8192, 128]" = torch.ops.aten.permute.default(view_967, [1, 0])
    mm_282: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_835, view_152);  permute_835 = view_152 = None
    permute_836: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_282, [1, 0]);  mm_282 = None
    sum_238: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
    view_968: "f32[8192]" = torch.ops.aten.view.default(sum_238, [8192]);  sum_238 = None
    permute_837: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
    view_969: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_281, [1, 128, 2048]);  mm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_638: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_969, primals_88);  primals_88 = None
    mul_639: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_638, 2048)
    sum_239: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True)
    mul_640: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_638, mul_50);  mul_638 = None
    sum_240: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [2], True);  mul_640 = None
    mul_641: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, sum_240);  sum_240 = None
    sub_217: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_639, sum_239);  mul_639 = sum_239 = None
    sub_218: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_217, mul_641);  sub_217 = mul_641 = None
    mul_642: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_61, sub_218);  div_61 = sub_218 = None
    mul_643: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_969, mul_50);  mul_50 = None
    sum_241: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
    sum_242: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_969, [0, 1]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_334: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_331, mul_642);  add_331 = mul_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_970: "f32[128, 2048]" = torch.ops.aten.view.default(add_334, [128, 2048])
    mm_283: "f32[128, 2048]" = torch.ops.aten.mm.default(view_970, permute_838);  permute_838 = None
    permute_839: "f32[2048, 128]" = torch.ops.aten.permute.default(view_970, [1, 0])
    mm_284: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_839, view_150);  permute_839 = view_150 = None
    permute_840: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_284, [1, 0]);  mm_284 = None
    sum_243: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_970, [0], True);  view_970 = None
    view_971: "f32[2048]" = torch.ops.aten.view.default(sum_243, [2048]);  sum_243 = None
    permute_841: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_840, [1, 0]);  permute_840 = None
    view_972: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_283, [1, 128, 2048]);  mm_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_973: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_972, [1, 128, 16, 128]);  view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_842: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_973, [0, 2, 1, 3]);  view_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_974: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_842, [16, 128, 128]);  permute_842 = None
    bmm_116: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_843, view_974);  permute_843 = None
    bmm_117: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_974, permute_844);  view_974 = permute_844 = None
    view_975: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_116, [1, 16, 128, 128]);  bmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_335: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_16, view_975);  tangents_16 = view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_976: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_644: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_976, alias_85);  view_976 = None
    sum_244: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [-1], True)
    mul_645: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_85, sum_244);  alias_85 = sum_244 = None
    sub_219: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_45: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, sub_219, full_default_25);  slice_28 = sub_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_977: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_45, [16, 128, 128]);  where_45 = None
    bmm_118: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_845, view_977);  permute_845 = None
    bmm_119: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_977, permute_846);  view_977 = permute_846 = None
    view_978: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_118, [1, 16, 128, 128]);  bmm_118 = None
    view_979: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_119, [1, 16, 128, 128]);  bmm_119 = None
    permute_847: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_978, [0, 1, 3, 2]);  view_978 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_336: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_15, permute_847);  tangents_15 = permute_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_848: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_335, [0, 2, 1, 3]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_148: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
    view_980: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_148, [1, 128, 2048]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_849: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_336, [0, 2, 1, 3]);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_149: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_849, memory_format = torch.contiguous_format);  permute_849 = None
    view_981: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_149, [1, 128, 2048]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_850: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_979, [0, 2, 1, 3]);  view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_150: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_850, memory_format = torch.contiguous_format);  permute_850 = None
    view_982: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_150, [1, 128, 2048]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_983: "f32[128, 2048]" = torch.ops.aten.view.default(view_980, [128, 2048]);  view_980 = None
    permute_851: "f32[2048, 128]" = torch.ops.aten.permute.default(view_983, [1, 0])
    mm_285: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_851, view_134);  permute_851 = None
    permute_852: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    mm_286: "f32[128, 2048]" = torch.ops.aten.mm.default(view_983, permute_853);  view_983 = permute_853 = None
    view_984: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_286, [1, 128, 2048]);  mm_286 = None
    permute_854: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_985: "f32[128, 2048]" = torch.ops.aten.view.default(view_981, [128, 2048]);  view_981 = None
    permute_855: "f32[2048, 128]" = torch.ops.aten.permute.default(view_985, [1, 0])
    mm_287: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_855, view_134);  permute_855 = None
    permute_856: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    mm_288: "f32[128, 2048]" = torch.ops.aten.mm.default(view_985, permute_857);  view_985 = permute_857 = None
    view_986: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_288, [1, 128, 2048]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_337: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_984, view_986);  view_984 = view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_858: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_856, [1, 0]);  permute_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_987: "f32[128, 2048]" = torch.ops.aten.view.default(view_982, [128, 2048]);  view_982 = None
    permute_859: "f32[2048, 128]" = torch.ops.aten.permute.default(view_987, [1, 0])
    mm_289: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_859, view_134);  permute_859 = view_134 = None
    permute_860: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    mm_290: "f32[128, 2048]" = torch.ops.aten.mm.default(view_987, permute_861);  view_987 = permute_861 = None
    view_988: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_290, [1, 128, 2048]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_338: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_337, view_988);  add_337 = view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_862: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_860, [1, 0]);  permute_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_647: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_338, primals_81);  primals_81 = None
    mul_648: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_647, 2048)
    sum_245: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True)
    mul_649: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_647, mul_48);  mul_647 = None
    sum_246: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True);  mul_649 = None
    mul_650: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, sum_246);  sum_246 = None
    sub_221: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_648, sum_245);  mul_648 = sum_245 = None
    sub_222: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_221, mul_650);  sub_221 = mul_650 = None
    mul_651: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_62, sub_222);  div_62 = sub_222 = None
    mul_652: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_338, mul_48);  mul_48 = None
    sum_247: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1]);  mul_652 = None
    sum_248: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_339: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_334, mul_651);  add_334 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_989: "f32[128, 2048]" = torch.ops.aten.view.default(add_339, [128, 2048])
    mm_291: "f32[128, 8192]" = torch.ops.aten.mm.default(view_989, permute_863);  permute_863 = None
    permute_864: "f32[2048, 128]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_292: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_864, view_132);  permute_864 = view_132 = None
    permute_865: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_292, [1, 0]);  mm_292 = None
    sum_249: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[2048]" = torch.ops.aten.view.default(sum_249, [2048]);  sum_249 = None
    permute_866: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_865, [1, 0]);  permute_865 = None
    view_991: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_291, [1, 128, 8192]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_653: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_991, mul_44);  mul_44 = None
    mul_654: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_991, add_47);  view_991 = add_47 = None
    alias_86: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_655: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_86, alias_86);  alias_86 = None
    sub_223: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_655);  mul_655 = None
    mul_656: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_653, sub_223);  mul_653 = sub_223 = None
    mul_657: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_656, 0.7978845608028654);  mul_656 = None
    mul_658: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_657, 0.044715)
    pow_43: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
    mul_659: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
    mul_660: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_658, mul_659);  mul_658 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_340: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_657, mul_660);  mul_657 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_661: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_654, 0.5);  mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_341: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_340, mul_661);  add_340 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_992: "f32[128, 8192]" = torch.ops.aten.view.default(add_341, [128, 8192]);  add_341 = None
    mm_293: "f32[128, 2048]" = torch.ops.aten.mm.default(view_992, permute_867);  permute_867 = None
    permute_868: "f32[8192, 128]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_294: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_868, view_130);  permute_868 = view_130 = None
    permute_869: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_294, [1, 0]);  mm_294 = None
    sum_250: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[8192]" = torch.ops.aten.view.default(sum_250, [8192]);  sum_250 = None
    permute_870: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_869, [1, 0]);  permute_869 = None
    view_994: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_293, [1, 128, 2048]);  mm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_663: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_994, primals_75);  primals_75 = None
    mul_664: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_663, 2048)
    sum_251: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
    mul_665: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_663, mul_42);  mul_663 = None
    sum_252: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
    mul_666: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, sum_252);  sum_252 = None
    sub_225: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_664, sum_251);  mul_664 = sum_251 = None
    sub_226: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_225, mul_666);  sub_225 = mul_666 = None
    mul_667: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_63, sub_226);  div_63 = sub_226 = None
    mul_668: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_994, mul_42);  mul_42 = None
    sum_253: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
    sum_254: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_342: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_339, mul_667);  add_339 = mul_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_995: "f32[128, 2048]" = torch.ops.aten.view.default(add_342, [128, 2048])
    mm_295: "f32[128, 2048]" = torch.ops.aten.mm.default(view_995, permute_871);  permute_871 = None
    permute_872: "f32[2048, 128]" = torch.ops.aten.permute.default(view_995, [1, 0])
    mm_296: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_872, view_128);  permute_872 = view_128 = None
    permute_873: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_296, [1, 0]);  mm_296 = None
    sum_255: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
    view_996: "f32[2048]" = torch.ops.aten.view.default(sum_255, [2048]);  sum_255 = None
    permute_874: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
    view_997: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_295, [1, 128, 2048]);  mm_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_998: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_997, [1, 128, 16, 128]);  view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_875: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_999: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_875, [16, 128, 128]);  permute_875 = None
    bmm_120: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_876, view_999);  permute_876 = None
    bmm_121: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_999, permute_877);  view_999 = permute_877 = None
    view_1000: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_120, [1, 16, 128, 128]);  bmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_343: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_14, view_1000);  tangents_14 = view_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1001: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_669: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1001, alias_87);  view_1001 = None
    sum_256: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [-1], True)
    mul_670: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_87, sum_256);  alias_87 = sum_256 = None
    sub_227: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_46: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, sub_227, full_default_25);  slice_24 = sub_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1002: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_46, [16, 128, 128]);  where_46 = None
    bmm_122: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_878, view_1002);  permute_878 = None
    bmm_123: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1002, permute_879);  view_1002 = permute_879 = None
    view_1003: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_122, [1, 16, 128, 128]);  bmm_122 = None
    view_1004: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_123, [1, 16, 128, 128]);  bmm_123 = None
    permute_880: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1003, [0, 1, 3, 2]);  view_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_344: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_13, permute_880);  tangents_13 = permute_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_881: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_343, [0, 2, 1, 3]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_151: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_1005: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_151, [1, 128, 2048]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_882: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_344, [0, 2, 1, 3]);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_152: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_882, memory_format = torch.contiguous_format);  permute_882 = None
    view_1006: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_152, [1, 128, 2048]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_883: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1004, [0, 2, 1, 3]);  view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_153: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_883, memory_format = torch.contiguous_format);  permute_883 = None
    view_1007: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_153, [1, 128, 2048]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1008: "f32[128, 2048]" = torch.ops.aten.view.default(view_1005, [128, 2048]);  view_1005 = None
    permute_884: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1008, [1, 0])
    mm_297: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_884, view_112);  permute_884 = None
    permute_885: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    mm_298: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1008, permute_886);  view_1008 = permute_886 = None
    view_1009: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_298, [1, 128, 2048]);  mm_298 = None
    permute_887: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_885, [1, 0]);  permute_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1010: "f32[128, 2048]" = torch.ops.aten.view.default(view_1006, [128, 2048]);  view_1006 = None
    permute_888: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1010, [1, 0])
    mm_299: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_888, view_112);  permute_888 = None
    permute_889: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_299, [1, 0]);  mm_299 = None
    mm_300: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1010, permute_890);  view_1010 = permute_890 = None
    view_1011: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_300, [1, 128, 2048]);  mm_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_345: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1009, view_1011);  view_1009 = view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_891: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_889, [1, 0]);  permute_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1012: "f32[128, 2048]" = torch.ops.aten.view.default(view_1007, [128, 2048]);  view_1007 = None
    permute_892: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1012, [1, 0])
    mm_301: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_892, view_112);  permute_892 = view_112 = None
    permute_893: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_301, [1, 0]);  mm_301 = None
    mm_302: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1012, permute_894);  view_1012 = permute_894 = None
    view_1013: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_302, [1, 128, 2048]);  mm_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_346: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_345, view_1013);  add_345 = view_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_895: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_893, [1, 0]);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_672: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_346, primals_68);  primals_68 = None
    mul_673: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_672, 2048)
    sum_257: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True)
    mul_674: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_672, mul_40);  mul_672 = None
    sum_258: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
    mul_675: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, sum_258);  sum_258 = None
    sub_229: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_673, sum_257);  mul_673 = sum_257 = None
    sub_230: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_229, mul_675);  sub_229 = mul_675 = None
    mul_676: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_64, sub_230);  div_64 = sub_230 = None
    mul_677: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_346, mul_40);  mul_40 = None
    sum_259: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1]);  mul_677 = None
    sum_260: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_347: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_342, mul_676);  add_342 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1014: "f32[128, 2048]" = torch.ops.aten.view.default(add_347, [128, 2048])
    mm_303: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1014, permute_896);  permute_896 = None
    permute_897: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_304: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_897, view_110);  permute_897 = view_110 = None
    permute_898: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_304, [1, 0]);  mm_304 = None
    sum_261: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[2048]" = torch.ops.aten.view.default(sum_261, [2048]);  sum_261 = None
    permute_899: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_1016: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_303, [1, 128, 8192]);  mm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_678: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1016, mul_36);  mul_36 = None
    mul_679: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1016, add_39);  view_1016 = add_39 = None
    alias_88: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_680: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_88, alias_88);  alias_88 = None
    sub_231: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_680);  mul_680 = None
    mul_681: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_678, sub_231);  mul_678 = sub_231 = None
    mul_682: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_681, 0.7978845608028654);  mul_681 = None
    mul_683: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_682, 0.044715)
    pow_44: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
    mul_684: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
    mul_685: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_348: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_682, mul_685);  mul_682 = mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_686: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_679, 0.5);  mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_349: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_348, mul_686);  add_348 = mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1017: "f32[128, 8192]" = torch.ops.aten.view.default(add_349, [128, 8192]);  add_349 = None
    mm_305: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1017, permute_900);  permute_900 = None
    permute_901: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1017, [1, 0])
    mm_306: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_901, view_108);  permute_901 = view_108 = None
    permute_902: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_306, [1, 0]);  mm_306 = None
    sum_262: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
    view_1018: "f32[8192]" = torch.ops.aten.view.default(sum_262, [8192]);  sum_262 = None
    permute_903: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_902, [1, 0]);  permute_902 = None
    view_1019: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_305, [1, 128, 2048]);  mm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_688: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1019, primals_62);  primals_62 = None
    mul_689: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_688, 2048)
    sum_263: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True)
    mul_690: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_688, mul_34);  mul_688 = None
    sum_264: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_690, [2], True);  mul_690 = None
    mul_691: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, sum_264);  sum_264 = None
    sub_233: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_689, sum_263);  mul_689 = sum_263 = None
    sub_234: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_233, mul_691);  sub_233 = mul_691 = None
    mul_692: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_65, sub_234);  div_65 = sub_234 = None
    mul_693: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1019, mul_34);  mul_34 = None
    sum_265: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 1]);  mul_693 = None
    sum_266: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1019, [0, 1]);  view_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_350: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_347, mul_692);  add_347 = mul_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1020: "f32[128, 2048]" = torch.ops.aten.view.default(add_350, [128, 2048])
    mm_307: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1020, permute_904);  permute_904 = None
    permute_905: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1020, [1, 0])
    mm_308: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_905, view_106);  permute_905 = view_106 = None
    permute_906: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_308, [1, 0]);  mm_308 = None
    sum_267: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
    view_1021: "f32[2048]" = torch.ops.aten.view.default(sum_267, [2048]);  sum_267 = None
    permute_907: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_906, [1, 0]);  permute_906 = None
    view_1022: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_307, [1, 128, 2048]);  mm_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1023: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1022, [1, 128, 16, 128]);  view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_908: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1024: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_908, [16, 128, 128]);  permute_908 = None
    bmm_124: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_909, view_1024);  permute_909 = None
    bmm_125: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1024, permute_910);  view_1024 = permute_910 = None
    view_1025: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_124, [1, 16, 128, 128]);  bmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_351: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_12, view_1025);  tangents_12 = view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1026: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_694: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1026, alias_89);  view_1026 = None
    sum_268: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [-1], True)
    mul_695: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_268);  alias_89 = sum_268 = None
    sub_235: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_47: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, sub_235, full_default_25);  slice_20 = sub_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1027: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_47, [16, 128, 128]);  where_47 = None
    bmm_126: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_911, view_1027);  permute_911 = None
    bmm_127: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1027, permute_912);  view_1027 = permute_912 = None
    view_1028: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_126, [1, 16, 128, 128]);  bmm_126 = None
    view_1029: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_127, [1, 16, 128, 128]);  bmm_127 = None
    permute_913: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1028, [0, 1, 3, 2]);  view_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_352: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_11, permute_913);  tangents_11 = permute_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_914: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_351, [0, 2, 1, 3]);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_154: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
    view_1030: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_154, [1, 128, 2048]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_915: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_352, [0, 2, 1, 3]);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_155: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_1031: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_155, [1, 128, 2048]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_916: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1029, [0, 2, 1, 3]);  view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_156: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
    view_1032: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_156, [1, 128, 2048]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1033: "f32[128, 2048]" = torch.ops.aten.view.default(view_1030, [128, 2048]);  view_1030 = None
    permute_917: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1033, [1, 0])
    mm_309: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_917, view_90);  permute_917 = None
    permute_918: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    mm_310: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1033, permute_919);  view_1033 = permute_919 = None
    view_1034: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_310, [1, 128, 2048]);  mm_310 = None
    permute_920: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_918, [1, 0]);  permute_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1035: "f32[128, 2048]" = torch.ops.aten.view.default(view_1031, [128, 2048]);  view_1031 = None
    permute_921: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1035, [1, 0])
    mm_311: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_921, view_90);  permute_921 = None
    permute_922: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_311, [1, 0]);  mm_311 = None
    mm_312: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1035, permute_923);  view_1035 = permute_923 = None
    view_1036: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_312, [1, 128, 2048]);  mm_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_353: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1034, view_1036);  view_1034 = view_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_924: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_922, [1, 0]);  permute_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1037: "f32[128, 2048]" = torch.ops.aten.view.default(view_1032, [128, 2048]);  view_1032 = None
    permute_925: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1037, [1, 0])
    mm_313: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_925, view_90);  permute_925 = view_90 = None
    permute_926: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_313, [1, 0]);  mm_313 = None
    mm_314: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1037, permute_927);  view_1037 = permute_927 = None
    view_1038: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_314, [1, 128, 2048]);  mm_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_354: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_353, view_1038);  add_353 = view_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_928: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_926, [1, 0]);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_697: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_354, primals_55);  primals_55 = None
    mul_698: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_697, 2048)
    sum_269: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_697, [2], True)
    mul_699: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_697, mul_32);  mul_697 = None
    sum_270: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True);  mul_699 = None
    mul_700: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, sum_270);  sum_270 = None
    sub_237: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_698, sum_269);  mul_698 = sum_269 = None
    sub_238: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_237, mul_700);  sub_237 = mul_700 = None
    mul_701: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_66, sub_238);  div_66 = sub_238 = None
    mul_702: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_354, mul_32);  mul_32 = None
    sum_271: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_702, [0, 1]);  mul_702 = None
    sum_272: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_355: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_350, mul_701);  add_350 = mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1039: "f32[128, 2048]" = torch.ops.aten.view.default(add_355, [128, 2048])
    mm_315: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1039, permute_929);  permute_929 = None
    permute_930: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1039, [1, 0])
    mm_316: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_930, view_88);  permute_930 = view_88 = None
    permute_931: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_316, [1, 0]);  mm_316 = None
    sum_273: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1039, [0], True);  view_1039 = None
    view_1040: "f32[2048]" = torch.ops.aten.view.default(sum_273, [2048]);  sum_273 = None
    permute_932: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_1041: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_315, [1, 128, 8192]);  mm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_703: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1041, mul_28);  mul_28 = None
    mul_704: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1041, add_31);  view_1041 = add_31 = None
    alias_90: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_705: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_90, alias_90);  alias_90 = None
    sub_239: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_705);  mul_705 = None
    mul_706: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_703, sub_239);  mul_703 = sub_239 = None
    mul_707: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_706, 0.7978845608028654);  mul_706 = None
    mul_708: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_707, 0.044715)
    pow_45: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
    mul_709: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
    mul_710: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_356: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_707, mul_710);  mul_707 = mul_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_711: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_704, 0.5);  mul_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_357: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_356, mul_711);  add_356 = mul_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1042: "f32[128, 8192]" = torch.ops.aten.view.default(add_357, [128, 8192]);  add_357 = None
    mm_317: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1042, permute_933);  permute_933 = None
    permute_934: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1042, [1, 0])
    mm_318: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_934, view_86);  permute_934 = view_86 = None
    permute_935: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_318, [1, 0]);  mm_318 = None
    sum_274: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
    view_1043: "f32[8192]" = torch.ops.aten.view.default(sum_274, [8192]);  sum_274 = None
    permute_936: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_1044: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_317, [1, 128, 2048]);  mm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_713: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1044, primals_49);  primals_49 = None
    mul_714: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_713, 2048)
    sum_275: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True)
    mul_715: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_713, mul_26);  mul_713 = None
    sum_276: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True);  mul_715 = None
    mul_716: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, sum_276);  sum_276 = None
    sub_241: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_714, sum_275);  mul_714 = sum_275 = None
    sub_242: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_241, mul_716);  sub_241 = mul_716 = None
    mul_717: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_67, sub_242);  div_67 = sub_242 = None
    mul_718: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1044, mul_26);  mul_26 = None
    sum_277: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 1]);  mul_718 = None
    sum_278: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1044, [0, 1]);  view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_358: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_355, mul_717);  add_355 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1045: "f32[128, 2048]" = torch.ops.aten.view.default(add_358, [128, 2048])
    mm_319: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1045, permute_937);  permute_937 = None
    permute_938: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1045, [1, 0])
    mm_320: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_938, view_84);  permute_938 = view_84 = None
    permute_939: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_320, [1, 0]);  mm_320 = None
    sum_279: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
    view_1046: "f32[2048]" = torch.ops.aten.view.default(sum_279, [2048]);  sum_279 = None
    permute_940: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_939, [1, 0]);  permute_939 = None
    view_1047: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_319, [1, 128, 2048]);  mm_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1048: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1047, [1, 128, 16, 128]);  view_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_941: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1049: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_941, [16, 128, 128]);  permute_941 = None
    bmm_128: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_942, view_1049);  permute_942 = None
    bmm_129: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1049, permute_943);  view_1049 = permute_943 = None
    view_1050: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_128, [1, 16, 128, 128]);  bmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_359: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_10, view_1050);  tangents_10 = view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1051: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_719: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1051, alias_91);  view_1051 = None
    sum_280: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [-1], True)
    mul_720: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_91, sum_280);  alias_91 = sum_280 = None
    sub_243: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_48: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, sub_243, full_default_25);  slice_16 = sub_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1052: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_48, [16, 128, 128]);  where_48 = None
    bmm_130: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_944, view_1052);  permute_944 = None
    bmm_131: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1052, permute_945);  view_1052 = permute_945 = None
    view_1053: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_130, [1, 16, 128, 128]);  bmm_130 = None
    view_1054: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_131, [1, 16, 128, 128]);  bmm_131 = None
    permute_946: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1053, [0, 1, 3, 2]);  view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_360: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_9, permute_946);  tangents_9 = permute_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_947: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_359, [0, 2, 1, 3]);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_157: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
    view_1055: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_157, [1, 128, 2048]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_948: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_360, [0, 2, 1, 3]);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_158: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
    view_1056: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_158, [1, 128, 2048]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_949: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_159: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
    view_1057: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_159, [1, 128, 2048]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1058: "f32[128, 2048]" = torch.ops.aten.view.default(view_1055, [128, 2048]);  view_1055 = None
    permute_950: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1058, [1, 0])
    mm_321: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_950, view_68);  permute_950 = None
    permute_951: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
    mm_322: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1058, permute_952);  view_1058 = permute_952 = None
    view_1059: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_322, [1, 128, 2048]);  mm_322 = None
    permute_953: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_951, [1, 0]);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1060: "f32[128, 2048]" = torch.ops.aten.view.default(view_1056, [128, 2048]);  view_1056 = None
    permute_954: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1060, [1, 0])
    mm_323: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_954, view_68);  permute_954 = None
    permute_955: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_323, [1, 0]);  mm_323 = None
    mm_324: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1060, permute_956);  view_1060 = permute_956 = None
    view_1061: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_324, [1, 128, 2048]);  mm_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_361: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1059, view_1061);  view_1059 = view_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_957: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_955, [1, 0]);  permute_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1062: "f32[128, 2048]" = torch.ops.aten.view.default(view_1057, [128, 2048]);  view_1057 = None
    permute_958: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_325: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_958, view_68);  permute_958 = view_68 = None
    permute_959: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
    mm_326: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1062, permute_960);  view_1062 = permute_960 = None
    view_1063: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_326, [1, 128, 2048]);  mm_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_362: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_361, view_1063);  add_361 = view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_961: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_959, [1, 0]);  permute_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_722: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_362, primals_42);  primals_42 = None
    mul_723: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_722, 2048)
    sum_281: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_722, [2], True)
    mul_724: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_722, mul_24);  mul_722 = None
    sum_282: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_724, [2], True);  mul_724 = None
    mul_725: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, sum_282);  sum_282 = None
    sub_245: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_723, sum_281);  mul_723 = sum_281 = None
    sub_246: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_245, mul_725);  sub_245 = mul_725 = None
    mul_726: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_68, sub_246);  div_68 = sub_246 = None
    mul_727: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_362, mul_24);  mul_24 = None
    sum_283: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 1]);  mul_727 = None
    sum_284: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 1]);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_363: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_358, mul_726);  add_358 = mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1064: "f32[128, 2048]" = torch.ops.aten.view.default(add_363, [128, 2048])
    mm_327: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1064, permute_962);  permute_962 = None
    permute_963: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1064, [1, 0])
    mm_328: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_963, view_66);  permute_963 = view_66 = None
    permute_964: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_328, [1, 0]);  mm_328 = None
    sum_285: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1064, [0], True);  view_1064 = None
    view_1065: "f32[2048]" = torch.ops.aten.view.default(sum_285, [2048]);  sum_285 = None
    permute_965: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_964, [1, 0]);  permute_964 = None
    view_1066: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_327, [1, 128, 8192]);  mm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_728: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1066, mul_20);  mul_20 = None
    mul_729: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1066, add_23);  view_1066 = add_23 = None
    alias_92: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_730: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_92, alias_92);  alias_92 = None
    sub_247: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_730);  mul_730 = None
    mul_731: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_728, sub_247);  mul_728 = sub_247 = None
    mul_732: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_731, 0.7978845608028654);  mul_731 = None
    mul_733: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_732, 0.044715)
    pow_46: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
    mul_734: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
    mul_735: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_733, mul_734);  mul_733 = mul_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_364: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_732, mul_735);  mul_732 = mul_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_736: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_729, 0.5);  mul_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_365: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_364, mul_736);  add_364 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1067: "f32[128, 8192]" = torch.ops.aten.view.default(add_365, [128, 8192]);  add_365 = None
    mm_329: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1067, permute_966);  permute_966 = None
    permute_967: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1067, [1, 0])
    mm_330: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_967, view_64);  permute_967 = view_64 = None
    permute_968: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_330, [1, 0]);  mm_330 = None
    sum_286: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1067, [0], True);  view_1067 = None
    view_1068: "f32[8192]" = torch.ops.aten.view.default(sum_286, [8192]);  sum_286 = None
    permute_969: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None
    view_1069: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_329, [1, 128, 2048]);  mm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_738: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1069, primals_36);  primals_36 = None
    mul_739: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_738, 2048)
    sum_287: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2], True)
    mul_740: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_738, mul_18);  mul_738 = None
    sum_288: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_740, [2], True);  mul_740 = None
    mul_741: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, sum_288);  sum_288 = None
    sub_249: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_739, sum_287);  mul_739 = sum_287 = None
    sub_250: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_249, mul_741);  sub_249 = mul_741 = None
    mul_742: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_69, sub_250);  div_69 = sub_250 = None
    mul_743: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1069, mul_18);  mul_18 = None
    sum_289: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 1]);  mul_743 = None
    sum_290: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1069, [0, 1]);  view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_366: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_363, mul_742);  add_363 = mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1070: "f32[128, 2048]" = torch.ops.aten.view.default(add_366, [128, 2048])
    mm_331: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1070, permute_970);  permute_970 = None
    permute_971: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1070, [1, 0])
    mm_332: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_971, view_62);  permute_971 = view_62 = None
    permute_972: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_332, [1, 0]);  mm_332 = None
    sum_291: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
    view_1071: "f32[2048]" = torch.ops.aten.view.default(sum_291, [2048]);  sum_291 = None
    permute_973: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_972, [1, 0]);  permute_972 = None
    view_1072: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_331, [1, 128, 2048]);  mm_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1073: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1072, [1, 128, 16, 128]);  view_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_974: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1074: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_974, [16, 128, 128]);  permute_974 = None
    bmm_132: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_975, view_1074);  permute_975 = None
    bmm_133: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1074, permute_976);  view_1074 = permute_976 = None
    view_1075: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_132, [1, 16, 128, 128]);  bmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_367: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_8, view_1075);  tangents_8 = view_1075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1076: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_744: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1076, alias_93);  view_1076 = None
    sum_292: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [-1], True)
    mul_745: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_93, sum_292);  alias_93 = sum_292 = None
    sub_251: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_49: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, sub_251, full_default_25);  slice_12 = sub_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1077: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_49, [16, 128, 128]);  where_49 = None
    bmm_134: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_977, view_1077);  permute_977 = None
    bmm_135: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1077, permute_978);  view_1077 = permute_978 = None
    view_1078: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_134, [1, 16, 128, 128]);  bmm_134 = None
    view_1079: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_135, [1, 16, 128, 128]);  bmm_135 = None
    permute_979: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1078, [0, 1, 3, 2]);  view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_368: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_7, permute_979);  tangents_7 = permute_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_980: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_367, [0, 2, 1, 3]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_160: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
    view_1080: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_160, [1, 128, 2048]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_981: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_368, [0, 2, 1, 3]);  add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_161: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_981, memory_format = torch.contiguous_format);  permute_981 = None
    view_1081: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_161, [1, 128, 2048]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_982: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1079, [0, 2, 1, 3]);  view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_162: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
    view_1082: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_162, [1, 128, 2048]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1083: "f32[128, 2048]" = torch.ops.aten.view.default(view_1080, [128, 2048]);  view_1080 = None
    permute_983: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1083, [1, 0])
    mm_333: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_983, view_46);  permute_983 = None
    permute_984: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
    mm_334: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1083, permute_985);  view_1083 = permute_985 = None
    view_1084: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_334, [1, 128, 2048]);  mm_334 = None
    permute_986: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_984, [1, 0]);  permute_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1085: "f32[128, 2048]" = torch.ops.aten.view.default(view_1081, [128, 2048]);  view_1081 = None
    permute_987: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1085, [1, 0])
    mm_335: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_987, view_46);  permute_987 = None
    permute_988: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_335, [1, 0]);  mm_335 = None
    mm_336: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1085, permute_989);  view_1085 = permute_989 = None
    view_1086: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_336, [1, 128, 2048]);  mm_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_369: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1084, view_1086);  view_1084 = view_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_990: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_988, [1, 0]);  permute_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1087: "f32[128, 2048]" = torch.ops.aten.view.default(view_1082, [128, 2048]);  view_1082 = None
    permute_991: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1087, [1, 0])
    mm_337: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_991, view_46);  permute_991 = view_46 = None
    permute_992: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_337, [1, 0]);  mm_337 = None
    mm_338: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1087, permute_993);  view_1087 = permute_993 = None
    view_1088: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_338, [1, 128, 2048]);  mm_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_370: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_369, view_1088);  add_369 = view_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_994: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_992, [1, 0]);  permute_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_747: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_370, primals_29);  primals_29 = None
    mul_748: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_747, 2048)
    sum_293: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
    mul_749: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_747, mul_16);  mul_747 = None
    sum_294: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
    mul_750: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, sum_294);  sum_294 = None
    sub_253: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_748, sum_293);  mul_748 = sum_293 = None
    sub_254: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_253, mul_750);  sub_253 = mul_750 = None
    mul_751: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_70, sub_254);  div_70 = sub_254 = None
    mul_752: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_370, mul_16);  mul_16 = None
    sum_295: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
    sum_296: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 1]);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_371: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_366, mul_751);  add_366 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1089: "f32[128, 2048]" = torch.ops.aten.view.default(add_371, [128, 2048])
    mm_339: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1089, permute_995);  permute_995 = None
    permute_996: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1089, [1, 0])
    mm_340: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_996, view_44);  permute_996 = view_44 = None
    permute_997: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_340, [1, 0]);  mm_340 = None
    sum_297: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
    view_1090: "f32[2048]" = torch.ops.aten.view.default(sum_297, [2048]);  sum_297 = None
    permute_998: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
    view_1091: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_339, [1, 128, 8192]);  mm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_753: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1091, mul_12);  mul_12 = None
    mul_754: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1091, add_15);  view_1091 = add_15 = None
    alias_94: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_755: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_94, alias_94);  alias_94 = None
    sub_255: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_755);  mul_755 = None
    mul_756: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_753, sub_255);  mul_753 = sub_255 = None
    mul_757: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_756, 0.7978845608028654);  mul_756 = None
    mul_758: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_757, 0.044715)
    pow_47: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_759: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
    mul_760: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_758, mul_759);  mul_758 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_372: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_757, mul_760);  mul_757 = mul_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_761: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_754, 0.5);  mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_373: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_372, mul_761);  add_372 = mul_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1092: "f32[128, 8192]" = torch.ops.aten.view.default(add_373, [128, 8192]);  add_373 = None
    mm_341: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1092, permute_999);  permute_999 = None
    permute_1000: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1092, [1, 0])
    mm_342: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1000, view_42);  permute_1000 = view_42 = None
    permute_1001: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_342, [1, 0]);  mm_342 = None
    sum_298: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1092, [0], True);  view_1092 = None
    view_1093: "f32[8192]" = torch.ops.aten.view.default(sum_298, [8192]);  sum_298 = None
    permute_1002: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_1001, [1, 0]);  permute_1001 = None
    view_1094: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_341, [1, 128, 2048]);  mm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_763: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1094, primals_23);  primals_23 = None
    mul_764: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_763, 2048)
    sum_299: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True)
    mul_765: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_763, mul_10);  mul_763 = None
    sum_300: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True);  mul_765 = None
    mul_766: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, sum_300);  sum_300 = None
    sub_257: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_764, sum_299);  mul_764 = sum_299 = None
    sub_258: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_257, mul_766);  sub_257 = mul_766 = None
    mul_767: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_71, sub_258);  div_71 = sub_258 = None
    mul_768: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1094, mul_10);  mul_10 = None
    sum_301: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 1]);  mul_768 = None
    sum_302: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1094, [0, 1]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_374: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_371, mul_767);  add_371 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1095: "f32[128, 2048]" = torch.ops.aten.view.default(add_374, [128, 2048])
    mm_343: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1095, permute_1003);  permute_1003 = None
    permute_1004: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1095, [1, 0])
    mm_344: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1004, view_40);  permute_1004 = view_40 = None
    permute_1005: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_344, [1, 0]);  mm_344 = None
    sum_303: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1095, [0], True);  view_1095 = None
    view_1096: "f32[2048]" = torch.ops.aten.view.default(sum_303, [2048]);  sum_303 = None
    permute_1006: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1005, [1, 0]);  permute_1005 = None
    view_1097: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_343, [1, 128, 2048]);  mm_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1098: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1097, [1, 128, 16, 128]);  view_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1007: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1098, [0, 2, 1, 3]);  view_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1099: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_1007, [16, 128, 128]);  permute_1007 = None
    bmm_136: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1008, view_1099);  permute_1008 = None
    bmm_137: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1099, permute_1009);  view_1099 = permute_1009 = None
    view_1100: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_136, [1, 16, 128, 128]);  bmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_375: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_6, view_1100);  tangents_6 = view_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1101: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_769: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1101, alias_95);  view_1101 = None
    sum_304: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [-1], True)
    mul_770: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_304);  alias_95 = sum_304 = None
    sub_259: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_50: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, sub_259, full_default_25);  slice_8 = sub_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1102: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_50, [16, 128, 128]);  where_50 = None
    bmm_138: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1010, view_1102);  permute_1010 = None
    bmm_139: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1102, permute_1011);  view_1102 = permute_1011 = None
    view_1103: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_138, [1, 16, 128, 128]);  bmm_138 = None
    view_1104: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_139, [1, 16, 128, 128]);  bmm_139 = None
    permute_1012: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1103, [0, 1, 3, 2]);  view_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_376: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_5, permute_1012);  tangents_5 = permute_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1013: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_375, [0, 2, 1, 3]);  add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_163: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
    view_1105: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_163, [1, 128, 2048]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1014: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_376, [0, 2, 1, 3]);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_164: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1014, memory_format = torch.contiguous_format);  permute_1014 = None
    view_1106: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_164, [1, 128, 2048]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1015: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1104, [0, 2, 1, 3]);  view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_165: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1015, memory_format = torch.contiguous_format);  permute_1015 = None
    view_1107: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_165, [1, 128, 2048]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1108: "f32[128, 2048]" = torch.ops.aten.view.default(view_1105, [128, 2048]);  view_1105 = None
    permute_1016: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1108, [1, 0])
    mm_345: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1016, view_24);  permute_1016 = None
    permute_1017: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
    mm_346: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1108, permute_1018);  view_1108 = permute_1018 = None
    view_1109: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_346, [1, 128, 2048]);  mm_346 = None
    permute_1019: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1017, [1, 0]);  permute_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1110: "f32[128, 2048]" = torch.ops.aten.view.default(view_1106, [128, 2048]);  view_1106 = None
    permute_1020: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1110, [1, 0])
    mm_347: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1020, view_24);  permute_1020 = None
    permute_1021: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_347, [1, 0]);  mm_347 = None
    mm_348: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1110, permute_1022);  view_1110 = permute_1022 = None
    view_1111: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_348, [1, 128, 2048]);  mm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_377: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1109, view_1111);  view_1109 = view_1111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1023: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1021, [1, 0]);  permute_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1112: "f32[128, 2048]" = torch.ops.aten.view.default(view_1107, [128, 2048]);  view_1107 = None
    permute_1024: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1112, [1, 0])
    mm_349: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1024, view_24);  permute_1024 = view_24 = None
    permute_1025: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_349, [1, 0]);  mm_349 = None
    mm_350: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1112, permute_1026);  view_1112 = permute_1026 = None
    view_1113: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_350, [1, 128, 2048]);  mm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_378: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_377, view_1113);  add_377 = view_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1027: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1025, [1, 0]);  permute_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_772: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_378, primals_16);  primals_16 = None
    mul_773: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_772, 2048)
    sum_305: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
    mul_774: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_772, mul_8);  mul_772 = None
    sum_306: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
    mul_775: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, sum_306);  sum_306 = None
    sub_261: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_773, sum_305);  mul_773 = sum_305 = None
    sub_262: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_261, mul_775);  sub_261 = mul_775 = None
    mul_776: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_72, sub_262);  div_72 = sub_262 = None
    mul_777: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_378, mul_8);  mul_8 = None
    sum_307: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
    sum_308: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_379: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_374, mul_776);  add_374 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1114: "f32[128, 2048]" = torch.ops.aten.view.default(add_379, [128, 2048])
    mm_351: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1114, permute_1028);  permute_1028 = None
    permute_1029: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1114, [1, 0])
    mm_352: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_1029, view_22);  permute_1029 = view_22 = None
    permute_1030: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_352, [1, 0]);  mm_352 = None
    sum_309: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1114, [0], True);  view_1114 = None
    view_1115: "f32[2048]" = torch.ops.aten.view.default(sum_309, [2048]);  sum_309 = None
    permute_1031: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_1030, [1, 0]);  permute_1030 = None
    view_1116: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_351, [1, 128, 8192]);  mm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_778: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1116, mul_4);  mul_4 = None
    mul_779: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1116, add_7);  view_1116 = add_7 = None
    alias_96: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_780: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_96, alias_96);  alias_96 = None
    sub_263: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_780);  mul_780 = None
    mul_781: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_778, sub_263);  mul_778 = sub_263 = None
    mul_782: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_781, 0.7978845608028654);  mul_781 = None
    mul_783: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_782, 0.044715)
    pow_48: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
    mul_784: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
    mul_785: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_380: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_782, mul_785);  mul_782 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_786: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_779, 0.5);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_381: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_380, mul_786);  add_380 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1117: "f32[128, 8192]" = torch.ops.aten.view.default(add_381, [128, 8192]);  add_381 = None
    mm_353: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1117, permute_1032);  permute_1032 = None
    permute_1033: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1117, [1, 0])
    mm_354: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1033, view_20);  permute_1033 = view_20 = None
    permute_1034: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_354, [1, 0]);  mm_354 = None
    sum_310: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1117, [0], True);  view_1117 = None
    view_1118: "f32[8192]" = torch.ops.aten.view.default(sum_310, [8192]);  sum_310 = None
    permute_1035: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_1034, [1, 0]);  permute_1034 = None
    view_1119: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_353, [1, 128, 2048]);  mm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    mul_788: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1119, primals_10);  primals_10 = None
    mul_789: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_788, 2048)
    sum_311: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [2], True)
    mul_790: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_788, mul_2);  mul_788 = None
    sum_312: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [2], True);  mul_790 = None
    mul_791: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, sum_312);  sum_312 = None
    sub_265: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_789, sum_311);  mul_789 = sum_311 = None
    sub_266: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_265, mul_791);  sub_265 = mul_791 = None
    mul_792: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_73, sub_266);  div_73 = sub_266 = None
    mul_793: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1119, mul_2);  mul_2 = None
    sum_313: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 1]);  mul_793 = None
    sum_314: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1119, [0, 1]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_382: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_379, mul_792);  add_379 = mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1120: "f32[128, 2048]" = torch.ops.aten.view.default(add_382, [128, 2048])
    mm_355: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1120, permute_1036);  permute_1036 = None
    permute_1037: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1120, [1, 0])
    mm_356: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1037, view_18);  permute_1037 = view_18 = None
    permute_1038: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_356, [1, 0]);  mm_356 = None
    sum_315: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1120, [0], True);  view_1120 = None
    view_1121: "f32[2048]" = torch.ops.aten.view.default(sum_315, [2048]);  sum_315 = None
    permute_1039: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1038, [1, 0]);  permute_1038 = None
    view_1122: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_355, [1, 128, 2048]);  mm_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1123: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1122, [1, 128, 16, 128]);  view_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1040: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1123, [0, 2, 1, 3]);  view_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1124: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_1040, [16, 128, 128]);  permute_1040 = None
    bmm_140: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1041, view_1124);  permute_1041 = None
    bmm_141: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1124, permute_1042);  view_1124 = permute_1042 = None
    view_1125: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_140, [1, 16, 128, 128]);  bmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_383: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_4, view_1125);  tangents_4 = view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1126: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_794: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1126, alias_97);  view_1126 = None
    sum_316: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_794, [-1], True)
    mul_795: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_97, sum_316);  alias_97 = sum_316 = None
    sub_267: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_794, mul_795);  mul_794 = mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_51: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, sub_267, full_default_25);  slice_4 = sub_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1127: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_51, [16, 128, 128]);  where_51 = None
    bmm_142: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1043, view_1127);  permute_1043 = None
    bmm_143: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1127, permute_1044);  view_1127 = permute_1044 = None
    view_1128: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_142, [1, 16, 128, 128]);  bmm_142 = None
    view_1129: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_143, [1, 16, 128, 128]);  bmm_143 = None
    permute_1045: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1128, [0, 1, 3, 2]);  view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_384: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_3, permute_1045);  tangents_3 = permute_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1046: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_383, [0, 2, 1, 3]);  add_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_166: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
    view_1130: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_166, [1, 128, 2048]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1047: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_384, [0, 2, 1, 3]);  add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_167: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1047, memory_format = torch.contiguous_format);  permute_1047 = None
    view_1131: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_167, [1, 128, 2048]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1048: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1129, [0, 2, 1, 3]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_168: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1048, memory_format = torch.contiguous_format);  permute_1048 = None
    view_1132: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_168, [1, 128, 2048]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1133: "f32[128, 2048]" = torch.ops.aten.view.default(view_1130, [128, 2048]);  view_1130 = None
    permute_1049: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1133, [1, 0])
    mm_357: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1049, view_2);  permute_1049 = None
    permute_1050: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
    mm_358: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1133, permute_1051);  view_1133 = permute_1051 = None
    view_1134: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_358, [1, 128, 2048]);  mm_358 = None
    permute_1052: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1050, [1, 0]);  permute_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1135: "f32[128, 2048]" = torch.ops.aten.view.default(view_1131, [128, 2048]);  view_1131 = None
    permute_1053: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1135, [1, 0])
    mm_359: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1053, view_2);  permute_1053 = None
    permute_1054: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_359, [1, 0]);  mm_359 = None
    mm_360: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1135, permute_1055);  view_1135 = permute_1055 = None
    view_1136: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_360, [1, 128, 2048]);  mm_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_385: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1134, view_1136);  view_1134 = view_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1056: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1054, [1, 0]);  permute_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1137: "f32[128, 2048]" = torch.ops.aten.view.default(view_1132, [128, 2048]);  view_1132 = None
    permute_1057: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1137, [1, 0])
    mm_361: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1057, view_2);  permute_1057 = view_2 = None
    permute_1058: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_361, [1, 0]);  mm_361 = None
    mm_362: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1137, permute_1059);  view_1137 = permute_1059 = None
    view_1138: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_362, [1, 128, 2048]);  mm_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_386: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_385, view_1138);  add_385 = view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1060: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1058, [1, 0]);  permute_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    mul_797: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_386, primals_3);  primals_3 = None
    mul_798: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_797, 2048)
    sum_317: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True)
    mul_799: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_797, mul);  mul_797 = None
    sum_318: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_799, [2], True);  mul_799 = None
    mul_800: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, sum_318);  sum_318 = None
    sub_269: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_798, sum_317);  mul_798 = sum_317 = None
    sub_270: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_269, mul_800);  sub_269 = mul_800 = None
    mul_801: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_74, sub_270);  div_74 = sub_270 = None
    mul_802: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_386, mul);  mul = None
    sum_319: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 1]);  mul_802 = None
    sum_320: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_386, [0, 1]);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_387: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_382, mul_801);  add_382 = mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    full_default_55: "b8[1, 128, 1]" = torch.ops.aten.full.default([1, 128, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_52: "f32[1, 128, 2048]" = torch.ops.aten.where.self(full_default_55, full_default_25, add_387);  full_default_55 = None
    full_default_57: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[2048, 2048]" = torch.ops.aten._unsafe_index_put.default(full_default_57, [view_1], where_52, True);  full_default_57 = view_1 = where_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_4: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_53: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_4, full_default_25, add_387);  unsqueeze_4 = full_default_25 = add_387 = None
    full_default_59: "f32[50257, 2048]" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 2048]" = torch.ops.aten._unsafe_index_put.default(full_default_59, [view], where_53, True);  full_default_59 = view = where_53 = None
    return [_unsafe_index_put_1, _unsafe_index_put, sum_319, sum_320, permute_1060, permute_1056, permute_1052, permute_1039, view_1121, sum_313, sum_314, permute_1035, view_1118, permute_1031, view_1115, sum_307, sum_308, permute_1027, permute_1023, permute_1019, permute_1006, view_1096, sum_301, sum_302, permute_1002, view_1093, permute_998, view_1090, sum_295, sum_296, permute_994, permute_990, permute_986, permute_973, view_1071, sum_289, sum_290, permute_969, view_1068, permute_965, view_1065, sum_283, sum_284, permute_961, permute_957, permute_953, permute_940, view_1046, sum_277, sum_278, permute_936, view_1043, permute_932, view_1040, sum_271, sum_272, permute_928, permute_924, permute_920, permute_907, view_1021, sum_265, sum_266, permute_903, view_1018, permute_899, view_1015, sum_259, sum_260, permute_895, permute_891, permute_887, permute_874, view_996, sum_253, sum_254, permute_870, view_993, permute_866, view_990, sum_247, sum_248, permute_862, permute_858, permute_854, permute_841, view_971, sum_241, sum_242, permute_837, view_968, permute_833, view_965, sum_235, sum_236, permute_829, permute_825, permute_821, permute_808, view_946, sum_229, sum_230, permute_804, view_943, permute_800, view_940, sum_223, sum_224, permute_796, permute_792, permute_788, permute_775, view_921, sum_217, sum_218, permute_771, view_918, permute_767, view_915, sum_211, sum_212, permute_763, permute_759, permute_755, permute_742, view_896, sum_205, sum_206, permute_738, view_893, permute_734, view_890, sum_199, sum_200, permute_730, permute_726, permute_722, permute_709, view_871, sum_193, sum_194, permute_705, view_868, permute_701, view_865, sum_187, sum_188, permute_697, permute_693, permute_689, permute_676, view_846, sum_181, sum_182, permute_672, view_843, permute_668, view_840, sum_175, sum_176, permute_664, permute_660, permute_656, permute_643, view_821, sum_169, sum_170, permute_639, view_818, permute_635, view_815, sum_163, sum_164, permute_631, permute_627, permute_623, permute_610, view_796, sum_157, sum_158, permute_606, view_793, permute_602, view_790, sum_151, sum_152, permute_598, permute_594, permute_590, permute_577, view_771, sum_145, sum_146, permute_573, view_768, permute_569, view_765, sum_139, sum_140, permute_565, permute_561, permute_557, permute_544, view_746, sum_133, sum_134, permute_540, view_743, permute_536, view_740, sum_127, sum_128, permute_532, permute_528, permute_524, permute_511, view_721, sum_121, sum_122, permute_507, view_718, permute_503, view_715, sum_115, sum_116, permute_499, permute_495, permute_491, permute_478, view_696, sum_109, sum_110, permute_474, view_693, permute_470, view_690, sum_103, sum_104, permute_466, permute_462, permute_458, permute_445, view_671, sum_97, sum_98, permute_441, view_668, permute_437, view_665, sum_91, sum_92, permute_433, permute_429, permute_425, permute_412, view_646, sum_85, sum_86, permute_408, view_643, permute_404, view_640, sum_79, sum_80, permute_400, permute_396, permute_392, permute_379, view_621, sum_73, sum_74, permute_375, view_618, permute_371, view_615, sum_67, sum_68, permute_367, permute_363, permute_359, permute_346, view_596, sum_61, sum_62, permute_342, view_593, permute_338, view_590, sum_55, sum_56, permute_334, permute_330, permute_326, permute_313, view_571, sum_49, sum_50, permute_309, view_568, permute_305, view_565, sum_43, sum_44, permute_301, permute_297, permute_293, permute_280, view_546, sum_37, sum_38, permute_276, view_543, permute_272, view_540, sum_31, sum_32, permute_268, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    