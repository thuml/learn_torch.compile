from __future__ import annotations



def forward(self, primals_2: "f32[4096]", primals_12: "f32[4096]", primals_22: "f32[4096]", primals_32: "f32[4096]", primals_42: "f32[4096]", primals_52: "f32[4096]", primals_62: "f32[4096]", primals_72: "f32[4096]", primals_82: "f32[4096]", primals_92: "f32[4096]", primals_102: "f32[4096]", primals_112: "f32[4096]", primals_122: "f32[4096]", primals_132: "f32[4096]", primals_142: "f32[4096]", primals_152: "f32[4096]", primals_162: "f32[4096]", primals_172: "f32[4096]", primals_182: "f32[4096]", primals_192: "f32[4096]", primals_202: "f32[4096]", primals_212: "f32[4096]", primals_222: "f32[4096]", primals_232: "f32[4096]", primals_242: "f32[4096]", primals_252: "f32[4096]", primals_262: "f32[4096]", primals_272: "f32[4096]", primals_282: "f32[4096]", primals_288: "f32[]", primals_291: "f32[]", primals_294: "f32[]", primals_297: "f32[]", primals_300: "f32[]", primals_303: "f32[]", primals_306: "f32[]", primals_309: "f32[]", primals_312: "f32[]", primals_315: "f32[]", primals_318: "f32[]", primals_321: "f32[]", primals_324: "f32[]", primals_327: "f32[]", primals_330: "f32[]", primals_333: "f32[]", primals_336: "f32[]", primals_339: "f32[]", primals_342: "f32[]", primals_345: "f32[]", primals_348: "f32[]", primals_351: "f32[]", primals_354: "f32[]", primals_357: "f32[]", primals_360: "f32[]", primals_363: "f32[]", primals_366: "f32[]", primals_369: "f32[]", primals_371: "i64[1, 128]", view: "i64[1, 128]", embedding: "f32[1, 128, 4096]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", view_2: "f32[128, 4096]", unsqueeze_3: "f32[1, 128, 1, 32, 1]", unsqueeze_5: "f32[1, 128, 1, 32, 1]", slice_48: "b8[1, 1, 128, 128]", view_24: "f32[128, 4096]", addmm: "f32[128, 16384]", tanh: "f32[1, 128, 16384]", view_28: "f32[128, 16384]", mul_10: "f32[1, 128, 4096]", view_30: "f32[128, 4096]", unsqueeze_16: "f32[1, 128, 1, 32, 1]", unsqueeze_18: "f32[1, 128, 1, 32, 1]", slice_96: "b8[1, 1, 128, 128]", view_52: "f32[128, 4096]", addmm_2: "f32[128, 16384]", tanh_1: "f32[1, 128, 16384]", view_56: "f32[128, 16384]", mul_20: "f32[1, 128, 4096]", view_58: "f32[128, 4096]", unsqueeze_29: "f32[1, 128, 1, 32, 1]", unsqueeze_31: "f32[1, 128, 1, 32, 1]", slice_144: "b8[1, 1, 128, 128]", view_80: "f32[128, 4096]", addmm_4: "f32[128, 16384]", tanh_2: "f32[1, 128, 16384]", view_84: "f32[128, 16384]", mul_30: "f32[1, 128, 4096]", view_86: "f32[128, 4096]", unsqueeze_42: "f32[1, 128, 1, 32, 1]", unsqueeze_44: "f32[1, 128, 1, 32, 1]", slice_192: "b8[1, 1, 128, 128]", view_108: "f32[128, 4096]", addmm_6: "f32[128, 16384]", tanh_3: "f32[1, 128, 16384]", view_112: "f32[128, 16384]", mul_40: "f32[1, 128, 4096]", view_114: "f32[128, 4096]", unsqueeze_55: "f32[1, 128, 1, 32, 1]", unsqueeze_57: "f32[1, 128, 1, 32, 1]", slice_240: "b8[1, 1, 128, 128]", view_136: "f32[128, 4096]", addmm_8: "f32[128, 16384]", tanh_4: "f32[1, 128, 16384]", view_140: "f32[128, 16384]", mul_50: "f32[1, 128, 4096]", view_142: "f32[128, 4096]", unsqueeze_68: "f32[1, 128, 1, 32, 1]", unsqueeze_70: "f32[1, 128, 1, 32, 1]", slice_288: "b8[1, 1, 128, 128]", view_164: "f32[128, 4096]", addmm_10: "f32[128, 16384]", tanh_5: "f32[1, 128, 16384]", view_168: "f32[128, 16384]", mul_60: "f32[1, 128, 4096]", view_170: "f32[128, 4096]", unsqueeze_81: "f32[1, 128, 1, 32, 1]", unsqueeze_83: "f32[1, 128, 1, 32, 1]", slice_336: "b8[1, 1, 128, 128]", view_192: "f32[128, 4096]", addmm_12: "f32[128, 16384]", tanh_6: "f32[1, 128, 16384]", view_196: "f32[128, 16384]", mul_70: "f32[1, 128, 4096]", view_198: "f32[128, 4096]", unsqueeze_94: "f32[1, 128, 1, 32, 1]", unsqueeze_96: "f32[1, 128, 1, 32, 1]", slice_384: "b8[1, 1, 128, 128]", view_220: "f32[128, 4096]", addmm_14: "f32[128, 16384]", tanh_7: "f32[1, 128, 16384]", view_224: "f32[128, 16384]", mul_80: "f32[1, 128, 4096]", view_226: "f32[128, 4096]", unsqueeze_107: "f32[1, 128, 1, 32, 1]", unsqueeze_109: "f32[1, 128, 1, 32, 1]", slice_432: "b8[1, 1, 128, 128]", view_248: "f32[128, 4096]", addmm_16: "f32[128, 16384]", tanh_8: "f32[1, 128, 16384]", view_252: "f32[128, 16384]", mul_90: "f32[1, 128, 4096]", view_254: "f32[128, 4096]", unsqueeze_120: "f32[1, 128, 1, 32, 1]", unsqueeze_122: "f32[1, 128, 1, 32, 1]", slice_480: "b8[1, 1, 128, 128]", view_276: "f32[128, 4096]", addmm_18: "f32[128, 16384]", tanh_9: "f32[1, 128, 16384]", view_280: "f32[128, 16384]", mul_100: "f32[1, 128, 4096]", view_282: "f32[128, 4096]", unsqueeze_133: "f32[1, 128, 1, 32, 1]", unsqueeze_135: "f32[1, 128, 1, 32, 1]", slice_528: "b8[1, 1, 128, 128]", view_304: "f32[128, 4096]", addmm_20: "f32[128, 16384]", tanh_10: "f32[1, 128, 16384]", view_308: "f32[128, 16384]", mul_110: "f32[1, 128, 4096]", view_310: "f32[128, 4096]", unsqueeze_146: "f32[1, 128, 1, 32, 1]", unsqueeze_148: "f32[1, 128, 1, 32, 1]", slice_576: "b8[1, 1, 128, 128]", view_332: "f32[128, 4096]", addmm_22: "f32[128, 16384]", tanh_11: "f32[1, 128, 16384]", view_336: "f32[128, 16384]", mul_120: "f32[1, 128, 4096]", view_338: "f32[128, 4096]", unsqueeze_159: "f32[1, 128, 1, 32, 1]", unsqueeze_161: "f32[1, 128, 1, 32, 1]", slice_624: "b8[1, 1, 128, 128]", view_360: "f32[128, 4096]", addmm_24: "f32[128, 16384]", tanh_12: "f32[1, 128, 16384]", view_364: "f32[128, 16384]", mul_130: "f32[1, 128, 4096]", view_366: "f32[128, 4096]", unsqueeze_172: "f32[1, 128, 1, 32, 1]", unsqueeze_174: "f32[1, 128, 1, 32, 1]", slice_672: "b8[1, 1, 128, 128]", view_388: "f32[128, 4096]", addmm_26: "f32[128, 16384]", tanh_13: "f32[1, 128, 16384]", view_392: "f32[128, 16384]", mul_140: "f32[1, 128, 4096]", view_394: "f32[128, 4096]", unsqueeze_185: "f32[1, 128, 1, 32, 1]", unsqueeze_187: "f32[1, 128, 1, 32, 1]", slice_720: "b8[1, 1, 128, 128]", view_416: "f32[128, 4096]", addmm_28: "f32[128, 16384]", tanh_14: "f32[1, 128, 16384]", view_420: "f32[128, 16384]", mul_150: "f32[1, 128, 4096]", view_422: "f32[128, 4096]", unsqueeze_198: "f32[1, 128, 1, 32, 1]", unsqueeze_200: "f32[1, 128, 1, 32, 1]", slice_768: "b8[1, 1, 128, 128]", view_444: "f32[128, 4096]", addmm_30: "f32[128, 16384]", tanh_15: "f32[1, 128, 16384]", view_448: "f32[128, 16384]", mul_160: "f32[1, 128, 4096]", view_450: "f32[128, 4096]", unsqueeze_211: "f32[1, 128, 1, 32, 1]", unsqueeze_213: "f32[1, 128, 1, 32, 1]", slice_816: "b8[1, 1, 128, 128]", view_472: "f32[128, 4096]", addmm_32: "f32[128, 16384]", tanh_16: "f32[1, 128, 16384]", view_476: "f32[128, 16384]", mul_170: "f32[1, 128, 4096]", view_478: "f32[128, 4096]", unsqueeze_224: "f32[1, 128, 1, 32, 1]", unsqueeze_226: "f32[1, 128, 1, 32, 1]", slice_864: "b8[1, 1, 128, 128]", view_500: "f32[128, 4096]", addmm_34: "f32[128, 16384]", tanh_17: "f32[1, 128, 16384]", view_504: "f32[128, 16384]", mul_180: "f32[1, 128, 4096]", view_506: "f32[128, 4096]", unsqueeze_237: "f32[1, 128, 1, 32, 1]", unsqueeze_239: "f32[1, 128, 1, 32, 1]", slice_912: "b8[1, 1, 128, 128]", view_528: "f32[128, 4096]", addmm_36: "f32[128, 16384]", tanh_18: "f32[1, 128, 16384]", view_532: "f32[128, 16384]", mul_190: "f32[1, 128, 4096]", view_534: "f32[128, 4096]", unsqueeze_250: "f32[1, 128, 1, 32, 1]", unsqueeze_252: "f32[1, 128, 1, 32, 1]", slice_960: "b8[1, 1, 128, 128]", view_556: "f32[128, 4096]", addmm_38: "f32[128, 16384]", tanh_19: "f32[1, 128, 16384]", view_560: "f32[128, 16384]", mul_200: "f32[1, 128, 4096]", view_562: "f32[128, 4096]", unsqueeze_263: "f32[1, 128, 1, 32, 1]", unsqueeze_265: "f32[1, 128, 1, 32, 1]", slice_1008: "b8[1, 1, 128, 128]", view_584: "f32[128, 4096]", addmm_40: "f32[128, 16384]", tanh_20: "f32[1, 128, 16384]", view_588: "f32[128, 16384]", mul_210: "f32[1, 128, 4096]", view_590: "f32[128, 4096]", unsqueeze_276: "f32[1, 128, 1, 32, 1]", unsqueeze_278: "f32[1, 128, 1, 32, 1]", slice_1056: "b8[1, 1, 128, 128]", view_612: "f32[128, 4096]", addmm_42: "f32[128, 16384]", tanh_21: "f32[1, 128, 16384]", view_616: "f32[128, 16384]", mul_220: "f32[1, 128, 4096]", view_618: "f32[128, 4096]", unsqueeze_289: "f32[1, 128, 1, 32, 1]", unsqueeze_291: "f32[1, 128, 1, 32, 1]", slice_1104: "b8[1, 1, 128, 128]", view_640: "f32[128, 4096]", addmm_44: "f32[128, 16384]", tanh_22: "f32[1, 128, 16384]", view_644: "f32[128, 16384]", mul_230: "f32[1, 128, 4096]", view_646: "f32[128, 4096]", unsqueeze_302: "f32[1, 128, 1, 32, 1]", unsqueeze_304: "f32[1, 128, 1, 32, 1]", slice_1152: "b8[1, 1, 128, 128]", view_668: "f32[128, 4096]", addmm_46: "f32[128, 16384]", tanh_23: "f32[1, 128, 16384]", view_672: "f32[128, 16384]", mul_240: "f32[1, 128, 4096]", view_674: "f32[128, 4096]", unsqueeze_315: "f32[1, 128, 1, 32, 1]", unsqueeze_317: "f32[1, 128, 1, 32, 1]", slice_1200: "b8[1, 1, 128, 128]", view_696: "f32[128, 4096]", addmm_48: "f32[128, 16384]", tanh_24: "f32[1, 128, 16384]", view_700: "f32[128, 16384]", mul_250: "f32[1, 128, 4096]", view_702: "f32[128, 4096]", unsqueeze_328: "f32[1, 128, 1, 32, 1]", unsqueeze_330: "f32[1, 128, 1, 32, 1]", slice_1248: "b8[1, 1, 128, 128]", view_724: "f32[128, 4096]", addmm_50: "f32[128, 16384]", tanh_25: "f32[1, 128, 16384]", view_728: "f32[128, 16384]", mul_260: "f32[1, 128, 4096]", view_730: "f32[128, 4096]", unsqueeze_341: "f32[1, 128, 1, 32, 1]", unsqueeze_343: "f32[1, 128, 1, 32, 1]", slice_1296: "b8[1, 1, 128, 128]", view_752: "f32[128, 4096]", addmm_52: "f32[128, 16384]", tanh_26: "f32[1, 128, 16384]", view_756: "f32[128, 16384]", mul_270: "f32[1, 128, 4096]", view_758: "f32[128, 4096]", unsqueeze_354: "f32[1, 128, 1, 32, 1]", unsqueeze_356: "f32[1, 128, 1, 32, 1]", slice_1344: "b8[1, 1, 128, 128]", view_780: "f32[128, 4096]", addmm_54: "f32[128, 16384]", tanh_27: "f32[1, 128, 16384]", view_784: "f32[128, 16384]", mul_280: "f32[1, 128, 4096]", view_787: "f32[128, 4096]", sub_58: "f32[127, 50400]", convert_element_type: "f32[]", permute_309: "f32[50400, 4096]", div_58: "f32[1, 128, 1]", permute_313: "f32[4096, 16384]", permute_317: "f32[16384, 4096]", permute_323: "f32[4096, 4096]", permute_326: "f32[16, 128, 128]", permute_327: "f32[16, 256, 128]", alias_59: "f32[1, 16, 128, 128]", permute_328: "f32[16, 256, 128]", permute_329: "f32[16, 128, 256]", permute_336: "f32[4096, 4096]", permute_340: "f32[4096, 4096]", permute_344: "f32[4096, 4096]", div_60: "f32[1, 128, 1]", permute_346: "f32[4096, 16384]", permute_350: "f32[16384, 4096]", permute_356: "f32[4096, 4096]", permute_359: "f32[16, 128, 128]", permute_360: "f32[16, 256, 128]", alias_61: "f32[1, 16, 128, 128]", permute_361: "f32[16, 256, 128]", permute_362: "f32[16, 128, 256]", permute_369: "f32[4096, 4096]", permute_373: "f32[4096, 4096]", permute_377: "f32[4096, 4096]", div_62: "f32[1, 128, 1]", permute_379: "f32[4096, 16384]", permute_383: "f32[16384, 4096]", permute_389: "f32[4096, 4096]", permute_392: "f32[16, 128, 128]", permute_393: "f32[16, 256, 128]", alias_63: "f32[1, 16, 128, 128]", permute_394: "f32[16, 256, 128]", permute_395: "f32[16, 128, 256]", permute_402: "f32[4096, 4096]", permute_406: "f32[4096, 4096]", permute_410: "f32[4096, 4096]", div_64: "f32[1, 128, 1]", permute_412: "f32[4096, 16384]", permute_416: "f32[16384, 4096]", permute_422: "f32[4096, 4096]", permute_425: "f32[16, 128, 128]", permute_426: "f32[16, 256, 128]", alias_65: "f32[1, 16, 128, 128]", permute_427: "f32[16, 256, 128]", permute_428: "f32[16, 128, 256]", permute_435: "f32[4096, 4096]", permute_439: "f32[4096, 4096]", permute_443: "f32[4096, 4096]", div_66: "f32[1, 128, 1]", permute_445: "f32[4096, 16384]", permute_449: "f32[16384, 4096]", permute_455: "f32[4096, 4096]", permute_458: "f32[16, 128, 128]", permute_459: "f32[16, 256, 128]", alias_67: "f32[1, 16, 128, 128]", permute_460: "f32[16, 256, 128]", permute_461: "f32[16, 128, 256]", permute_468: "f32[4096, 4096]", permute_472: "f32[4096, 4096]", permute_476: "f32[4096, 4096]", div_68: "f32[1, 128, 1]", permute_478: "f32[4096, 16384]", permute_482: "f32[16384, 4096]", permute_488: "f32[4096, 4096]", permute_491: "f32[16, 128, 128]", permute_492: "f32[16, 256, 128]", alias_69: "f32[1, 16, 128, 128]", permute_493: "f32[16, 256, 128]", permute_494: "f32[16, 128, 256]", permute_501: "f32[4096, 4096]", permute_505: "f32[4096, 4096]", permute_509: "f32[4096, 4096]", div_70: "f32[1, 128, 1]", permute_511: "f32[4096, 16384]", permute_515: "f32[16384, 4096]", permute_521: "f32[4096, 4096]", permute_524: "f32[16, 128, 128]", permute_525: "f32[16, 256, 128]", alias_71: "f32[1, 16, 128, 128]", permute_526: "f32[16, 256, 128]", permute_527: "f32[16, 128, 256]", permute_534: "f32[4096, 4096]", permute_538: "f32[4096, 4096]", permute_542: "f32[4096, 4096]", div_72: "f32[1, 128, 1]", permute_544: "f32[4096, 16384]", permute_548: "f32[16384, 4096]", permute_554: "f32[4096, 4096]", permute_557: "f32[16, 128, 128]", permute_558: "f32[16, 256, 128]", alias_73: "f32[1, 16, 128, 128]", permute_559: "f32[16, 256, 128]", permute_560: "f32[16, 128, 256]", permute_567: "f32[4096, 4096]", permute_571: "f32[4096, 4096]", permute_575: "f32[4096, 4096]", div_74: "f32[1, 128, 1]", permute_577: "f32[4096, 16384]", permute_581: "f32[16384, 4096]", permute_587: "f32[4096, 4096]", permute_590: "f32[16, 128, 128]", permute_591: "f32[16, 256, 128]", alias_75: "f32[1, 16, 128, 128]", permute_592: "f32[16, 256, 128]", permute_593: "f32[16, 128, 256]", permute_600: "f32[4096, 4096]", permute_604: "f32[4096, 4096]", permute_608: "f32[4096, 4096]", div_76: "f32[1, 128, 1]", permute_610: "f32[4096, 16384]", permute_614: "f32[16384, 4096]", permute_620: "f32[4096, 4096]", permute_623: "f32[16, 128, 128]", permute_624: "f32[16, 256, 128]", alias_77: "f32[1, 16, 128, 128]", permute_625: "f32[16, 256, 128]", permute_626: "f32[16, 128, 256]", permute_633: "f32[4096, 4096]", permute_637: "f32[4096, 4096]", permute_641: "f32[4096, 4096]", div_78: "f32[1, 128, 1]", permute_643: "f32[4096, 16384]", permute_647: "f32[16384, 4096]", permute_653: "f32[4096, 4096]", permute_656: "f32[16, 128, 128]", permute_657: "f32[16, 256, 128]", alias_79: "f32[1, 16, 128, 128]", permute_658: "f32[16, 256, 128]", permute_659: "f32[16, 128, 256]", permute_666: "f32[4096, 4096]", permute_670: "f32[4096, 4096]", permute_674: "f32[4096, 4096]", div_80: "f32[1, 128, 1]", permute_676: "f32[4096, 16384]", permute_680: "f32[16384, 4096]", permute_686: "f32[4096, 4096]", permute_689: "f32[16, 128, 128]", permute_690: "f32[16, 256, 128]", alias_81: "f32[1, 16, 128, 128]", permute_691: "f32[16, 256, 128]", permute_692: "f32[16, 128, 256]", permute_699: "f32[4096, 4096]", permute_703: "f32[4096, 4096]", permute_707: "f32[4096, 4096]", div_82: "f32[1, 128, 1]", permute_709: "f32[4096, 16384]", permute_713: "f32[16384, 4096]", permute_719: "f32[4096, 4096]", permute_722: "f32[16, 128, 128]", permute_723: "f32[16, 256, 128]", alias_83: "f32[1, 16, 128, 128]", permute_724: "f32[16, 256, 128]", permute_725: "f32[16, 128, 256]", permute_732: "f32[4096, 4096]", permute_736: "f32[4096, 4096]", permute_740: "f32[4096, 4096]", div_84: "f32[1, 128, 1]", permute_742: "f32[4096, 16384]", permute_746: "f32[16384, 4096]", permute_752: "f32[4096, 4096]", permute_755: "f32[16, 128, 128]", permute_756: "f32[16, 256, 128]", alias_85: "f32[1, 16, 128, 128]", permute_757: "f32[16, 256, 128]", permute_758: "f32[16, 128, 256]", permute_765: "f32[4096, 4096]", permute_769: "f32[4096, 4096]", permute_773: "f32[4096, 4096]", div_86: "f32[1, 128, 1]", permute_775: "f32[4096, 16384]", permute_779: "f32[16384, 4096]", permute_785: "f32[4096, 4096]", permute_788: "f32[16, 128, 128]", permute_789: "f32[16, 256, 128]", alias_87: "f32[1, 16, 128, 128]", permute_790: "f32[16, 256, 128]", permute_791: "f32[16, 128, 256]", permute_798: "f32[4096, 4096]", permute_802: "f32[4096, 4096]", permute_806: "f32[4096, 4096]", div_88: "f32[1, 128, 1]", permute_808: "f32[4096, 16384]", permute_812: "f32[16384, 4096]", permute_818: "f32[4096, 4096]", permute_821: "f32[16, 128, 128]", permute_822: "f32[16, 256, 128]", alias_89: "f32[1, 16, 128, 128]", permute_823: "f32[16, 256, 128]", permute_824: "f32[16, 128, 256]", permute_831: "f32[4096, 4096]", permute_835: "f32[4096, 4096]", permute_839: "f32[4096, 4096]", div_90: "f32[1, 128, 1]", permute_841: "f32[4096, 16384]", permute_845: "f32[16384, 4096]", permute_851: "f32[4096, 4096]", permute_854: "f32[16, 128, 128]", permute_855: "f32[16, 256, 128]", alias_91: "f32[1, 16, 128, 128]", permute_856: "f32[16, 256, 128]", permute_857: "f32[16, 128, 256]", permute_864: "f32[4096, 4096]", permute_868: "f32[4096, 4096]", permute_872: "f32[4096, 4096]", div_92: "f32[1, 128, 1]", permute_874: "f32[4096, 16384]", permute_878: "f32[16384, 4096]", permute_884: "f32[4096, 4096]", permute_887: "f32[16, 128, 128]", permute_888: "f32[16, 256, 128]", alias_93: "f32[1, 16, 128, 128]", permute_889: "f32[16, 256, 128]", permute_890: "f32[16, 128, 256]", permute_897: "f32[4096, 4096]", permute_901: "f32[4096, 4096]", permute_905: "f32[4096, 4096]", div_94: "f32[1, 128, 1]", permute_907: "f32[4096, 16384]", permute_911: "f32[16384, 4096]", permute_917: "f32[4096, 4096]", permute_920: "f32[16, 128, 128]", permute_921: "f32[16, 256, 128]", alias_95: "f32[1, 16, 128, 128]", permute_922: "f32[16, 256, 128]", permute_923: "f32[16, 128, 256]", permute_930: "f32[4096, 4096]", permute_934: "f32[4096, 4096]", permute_938: "f32[4096, 4096]", div_96: "f32[1, 128, 1]", permute_940: "f32[4096, 16384]", permute_944: "f32[16384, 4096]", permute_950: "f32[4096, 4096]", permute_953: "f32[16, 128, 128]", permute_954: "f32[16, 256, 128]", alias_97: "f32[1, 16, 128, 128]", permute_955: "f32[16, 256, 128]", permute_956: "f32[16, 128, 256]", permute_963: "f32[4096, 4096]", permute_967: "f32[4096, 4096]", permute_971: "f32[4096, 4096]", div_98: "f32[1, 128, 1]", permute_973: "f32[4096, 16384]", permute_977: "f32[16384, 4096]", permute_983: "f32[4096, 4096]", permute_986: "f32[16, 128, 128]", permute_987: "f32[16, 256, 128]", alias_99: "f32[1, 16, 128, 128]", permute_988: "f32[16, 256, 128]", permute_989: "f32[16, 128, 256]", permute_996: "f32[4096, 4096]", permute_1000: "f32[4096, 4096]", permute_1004: "f32[4096, 4096]", div_100: "f32[1, 128, 1]", permute_1006: "f32[4096, 16384]", permute_1010: "f32[16384, 4096]", permute_1016: "f32[4096, 4096]", permute_1019: "f32[16, 128, 128]", permute_1020: "f32[16, 256, 128]", alias_101: "f32[1, 16, 128, 128]", permute_1021: "f32[16, 256, 128]", permute_1022: "f32[16, 128, 256]", permute_1029: "f32[4096, 4096]", permute_1033: "f32[4096, 4096]", permute_1037: "f32[4096, 4096]", div_102: "f32[1, 128, 1]", permute_1039: "f32[4096, 16384]", permute_1043: "f32[16384, 4096]", permute_1049: "f32[4096, 4096]", permute_1052: "f32[16, 128, 128]", permute_1053: "f32[16, 256, 128]", alias_103: "f32[1, 16, 128, 128]", permute_1054: "f32[16, 256, 128]", permute_1055: "f32[16, 128, 256]", permute_1062: "f32[4096, 4096]", permute_1066: "f32[4096, 4096]", permute_1070: "f32[4096, 4096]", div_104: "f32[1, 128, 1]", permute_1072: "f32[4096, 16384]", permute_1076: "f32[16384, 4096]", permute_1082: "f32[4096, 4096]", permute_1085: "f32[16, 128, 128]", permute_1086: "f32[16, 256, 128]", alias_105: "f32[1, 16, 128, 128]", permute_1087: "f32[16, 256, 128]", permute_1088: "f32[16, 128, 256]", permute_1095: "f32[4096, 4096]", permute_1099: "f32[4096, 4096]", permute_1103: "f32[4096, 4096]", div_106: "f32[1, 128, 1]", permute_1105: "f32[4096, 16384]", permute_1109: "f32[16384, 4096]", permute_1115: "f32[4096, 4096]", permute_1118: "f32[16, 128, 128]", permute_1119: "f32[16, 256, 128]", alias_107: "f32[1, 16, 128, 128]", permute_1120: "f32[16, 256, 128]", permute_1121: "f32[16, 128, 256]", permute_1128: "f32[4096, 4096]", permute_1132: "f32[4096, 4096]", permute_1136: "f32[4096, 4096]", div_108: "f32[1, 128, 1]", permute_1138: "f32[4096, 16384]", permute_1142: "f32[16384, 4096]", permute_1148: "f32[4096, 4096]", permute_1151: "f32[16, 128, 128]", permute_1152: "f32[16, 256, 128]", alias_109: "f32[1, 16, 128, 128]", permute_1153: "f32[16, 256, 128]", permute_1154: "f32[16, 128, 256]", permute_1161: "f32[4096, 4096]", permute_1165: "f32[4096, 4096]", permute_1169: "f32[4096, 4096]", div_110: "f32[1, 128, 1]", permute_1171: "f32[4096, 16384]", permute_1175: "f32[16384, 4096]", permute_1181: "f32[4096, 4096]", permute_1184: "f32[16, 128, 128]", permute_1185: "f32[16, 256, 128]", alias_111: "f32[1, 16, 128, 128]", permute_1186: "f32[16, 256, 128]", permute_1187: "f32[16, 128, 256]", permute_1194: "f32[4096, 4096]", permute_1198: "f32[4096, 4096]", permute_1202: "f32[4096, 4096]", div_112: "f32[1, 128, 1]", permute_1204: "f32[4096, 16384]", permute_1208: "f32[16384, 4096]", permute_1214: "f32[4096, 4096]", permute_1217: "f32[16, 128, 128]", permute_1218: "f32[16, 256, 128]", alias_113: "f32[1, 16, 128, 128]", permute_1219: "f32[16, 256, 128]", permute_1220: "f32[16, 128, 256]", permute_1227: "f32[4096, 4096]", permute_1231: "f32[4096, 4096]", permute_1235: "f32[4096, 4096]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 50400]", tangents_3: "f32[1, 16, 128, 256]", tangents_4: "f32[1, 16, 128, 256]", tangents_5: "f32[1, 16, 128, 256]", tangents_6: "f32[1, 16, 128, 256]", tangents_7: "f32[1, 16, 128, 256]", tangents_8: "f32[1, 16, 128, 256]", tangents_9: "f32[1, 16, 128, 256]", tangents_10: "f32[1, 16, 128, 256]", tangents_11: "f32[1, 16, 128, 256]", tangents_12: "f32[1, 16, 128, 256]", tangents_13: "f32[1, 16, 128, 256]", tangents_14: "f32[1, 16, 128, 256]", tangents_15: "f32[1, 16, 128, 256]", tangents_16: "f32[1, 16, 128, 256]", tangents_17: "f32[1, 16, 128, 256]", tangents_18: "f32[1, 16, 128, 256]", tangents_19: "f32[1, 16, 128, 256]", tangents_20: "f32[1, 16, 128, 256]", tangents_21: "f32[1, 16, 128, 256]", tangents_22: "f32[1, 16, 128, 256]", tangents_23: "f32[1, 16, 128, 256]", tangents_24: "f32[1, 16, 128, 256]", tangents_25: "f32[1, 16, 128, 256]", tangents_26: "f32[1, 16, 128, 256]", tangents_27: "f32[1, 16, 128, 256]", tangents_28: "f32[1, 16, 128, 256]", tangents_29: "f32[1, 16, 128, 256]", tangents_30: "f32[1, 16, 128, 256]", tangents_31: "f32[1, 16, 128, 256]", tangents_32: "f32[1, 16, 128, 256]", tangents_33: "f32[1, 16, 128, 256]", tangents_34: "f32[1, 16, 128, 256]", tangents_35: "f32[1, 16, 128, 256]", tangents_36: "f32[1, 16, 128, 256]", tangents_37: "f32[1, 16, 128, 256]", tangents_38: "f32[1, 16, 128, 256]", tangents_39: "f32[1, 16, 128, 256]", tangents_40: "f32[1, 16, 128, 256]", tangents_41: "f32[1, 16, 128, 256]", tangents_42: "f32[1, 16, 128, 256]", tangents_43: "f32[1, 16, 128, 256]", tangents_44: "f32[1, 16, 128, 256]", tangents_45: "f32[1, 16, 128, 256]", tangents_46: "f32[1, 16, 128, 256]", tangents_47: "f32[1, 16, 128, 256]", tangents_48: "f32[1, 16, 128, 256]", tangents_49: "f32[1, 16, 128, 256]", tangents_50: "f32[1, 16, 128, 256]", tangents_51: "f32[1, 16, 128, 256]", tangents_52: "f32[1, 16, 128, 256]", tangents_53: "f32[1, 16, 128, 256]", tangents_54: "f32[1, 16, 128, 256]", tangents_55: "f32[1, 16, 128, 256]", tangents_56: "f32[1, 16, 128, 256]", tangents_57: "f32[1, 16, 128, 256]", tangents_58: "f32[1, 16, 128, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
    mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 128, 1, 32, 2]);  unsqueeze_3 = None
    clone_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_11: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_5, [1, 128, 1, 32, 2]);  unsqueeze_5 = None
    clone_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_27: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    add_5: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_8: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_16, [1, 128, 1, 32, 2]);  unsqueeze_16 = None
    clone_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_39: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_18, [1, 128, 1, 32, 2]);  unsqueeze_18 = None
    clone_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_40: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_55: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_55, 0.5)
    add_13: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_16: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_29, [1, 128, 1, 32, 2]);  unsqueeze_29 = None
    clone_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_67: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_31, [1, 128, 1, 32, 2]);  unsqueeze_31 = None
    clone_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_68: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_83: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_26: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_83, 0.5)
    add_21: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_24: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_42, [1, 128, 1, 32, 2]);  unsqueeze_42 = None
    clone_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_95: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_44, [1, 128, 1, 32, 2]);  unsqueeze_44 = None
    clone_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_96: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_111: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    add_29: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_32: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_55, [1, 128, 1, 32, 2]);  unsqueeze_55 = None
    clone_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_123: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_57, [1, 128, 1, 32, 2]);  unsqueeze_57 = None
    clone_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_124: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_139: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_46: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
    add_37: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_40: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_68, [1, 128, 1, 32, 2]);  unsqueeze_68 = None
    clone_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_151: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_70, [1, 128, 1, 32, 2]);  unsqueeze_70 = None
    clone_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_152: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_167: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_48: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_81, [1, 128, 1, 32, 2]);  unsqueeze_81 = None
    clone_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_179: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_83, [1, 128, 1, 32, 2]);  unsqueeze_83 = None
    clone_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_180: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_195: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    add_53: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_56: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_94, [1, 128, 1, 32, 2]);  unsqueeze_94 = None
    clone_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_207: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_96, [1, 128, 1, 32, 2]);  unsqueeze_96 = None
    clone_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_208: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_223: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_223, 0.5)
    add_61: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_64: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_107, [1, 128, 1, 32, 2]);  unsqueeze_107 = None
    clone_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_235: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_109, [1, 128, 1, 32, 2]);  unsqueeze_109 = None
    clone_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_236: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_251: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    add_69: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_72: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_120, [1, 128, 1, 32, 2]);  unsqueeze_120 = None
    clone_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_263: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_122, [1, 128, 1, 32, 2]);  unsqueeze_122 = None
    clone_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_264: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_279: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_96: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    add_77: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_80: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_133, [1, 128, 1, 32, 2]);  unsqueeze_133 = None
    clone_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_291: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_135, [1, 128, 1, 32, 2]);  unsqueeze_135 = None
    clone_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_292: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_307: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    add_85: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_88: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_146, [1, 128, 1, 32, 2]);  unsqueeze_146 = None
    clone_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_319: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_148, [1, 128, 1, 32, 2]);  unsqueeze_148 = None
    clone_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_320: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_335: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_335, 0.5)
    add_93: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_96: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_159, [1, 128, 1, 32, 2]);  unsqueeze_159 = None
    clone_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_347: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_161, [1, 128, 1, 32, 2]);  unsqueeze_161 = None
    clone_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
    view_348: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_363: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_363, 0.5)
    add_101: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_104: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_172, [1, 128, 1, 32, 2]);  unsqueeze_172 = None
    clone_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_375: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_174, [1, 128, 1, 32, 2]);  unsqueeze_174 = None
    clone_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_376: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_391: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_136: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    add_109: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_112: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_185, [1, 128, 1, 32, 2]);  unsqueeze_185 = None
    clone_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
    view_403: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_187, [1, 128, 1, 32, 2]);  unsqueeze_187 = None
    clone_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_404: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_419: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_146: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_419, 0.5)
    add_117: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_120: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_198, [1, 128, 1, 32, 2]);  unsqueeze_198 = None
    clone_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
    view_431: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_200, [1, 128, 1, 32, 2]);  unsqueeze_200 = None
    clone_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
    view_432: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_447: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
    add_125: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_128: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_211, [1, 128, 1, 32, 2]);  unsqueeze_211 = None
    clone_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_459: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_213, [1, 128, 1, 32, 2]);  unsqueeze_213 = None
    clone_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_460: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_475: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_166: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_475, 0.5)
    add_133: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_136: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_224, [1, 128, 1, 32, 2]);  unsqueeze_224 = None
    clone_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
    view_487: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_226, [1, 128, 1, 32, 2]);  unsqueeze_226 = None
    clone_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_488: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_503: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    add_141: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_144: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_237, [1, 128, 1, 32, 2]);  unsqueeze_237 = None
    clone_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
    view_515: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_239, [1, 128, 1, 32, 2]);  unsqueeze_239 = None
    clone_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
    view_516: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_531: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_186: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_531, 0.5)
    add_149: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_152: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_250, [1, 128, 1, 32, 2]);  unsqueeze_250 = None
    clone_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
    view_543: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_252, [1, 128, 1, 32, 2]);  unsqueeze_252 = None
    clone_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
    view_544: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_559: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_559, 0.5)
    add_157: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_160: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_263, [1, 128, 1, 32, 2]);  unsqueeze_263 = None
    clone_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
    view_571: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_265, [1, 128, 1, 32, 2]);  unsqueeze_265 = None
    clone_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
    view_572: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_587: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_206: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_587, 0.5)
    add_165: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_168: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_276, [1, 128, 1, 32, 2]);  unsqueeze_276 = None
    clone_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
    view_599: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_278, [1, 128, 1, 32, 2]);  unsqueeze_278 = None
    clone_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
    view_600: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_615: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_615, 0.5)
    add_173: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_176: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_289, [1, 128, 1, 32, 2]);  unsqueeze_289 = None
    clone_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
    view_627: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_291, [1, 128, 1, 32, 2]);  unsqueeze_291 = None
    clone_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
    view_628: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_643: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
    add_181: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_184: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_302, [1, 128, 1, 32, 2]);  unsqueeze_302 = None
    clone_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
    view_655: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_304, [1, 128, 1, 32, 2]);  unsqueeze_304 = None
    clone_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
    view_656: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_671: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_671, 0.5)
    add_189: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_192: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_315, [1, 128, 1, 32, 2]);  unsqueeze_315 = None
    clone_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
    view_683: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_317, [1, 128, 1, 32, 2]);  unsqueeze_317 = None
    clone_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
    view_684: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_699: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_699, 0.5)
    add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_200: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_328, [1, 128, 1, 32, 2]);  unsqueeze_328 = None
    clone_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_200, memory_format = torch.contiguous_format);  expand_200 = None
    view_711: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_330, [1, 128, 1, 32, 2]);  unsqueeze_330 = None
    clone_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
    view_712: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_727: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_256: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_727, 0.5)
    add_205: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_208: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_341, [1, 128, 1, 32, 2]);  unsqueeze_341 = None
    clone_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
    view_739: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_343, [1, 128, 1, 32, 2]);  unsqueeze_343 = None
    clone_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
    view_740: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_755: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_755, 0.5)
    add_213: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    expand_216: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_354, [1, 128, 1, 32, 2]);  unsqueeze_354 = None
    clone_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
    view_767: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    expand_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_356, [1, 128, 1, 32, 2]);  unsqueeze_356 = None
    clone_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
    view_768: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_783: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_276: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_783, 0.5)
    add_221: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:886, code: shift_labels = labels[..., 1:].contiguous()
    slice_1347: "i64[1, 127]" = torch.ops.aten.slice.Tensor(primals_371, 1, 1, 9223372036854775807);  primals_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:889, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_790: "i64[127]" = torch.ops.aten.reshape.default(slice_1347, [-1]);  slice_1347 = None
    full_default_28: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_57: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_366: "i64[127, 1]" = torch.ops.aten.unsqueeze.default(view_790, 1);  view_790 = None
    ne_3: "b8[127, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_366, -100)
    where_30: "i64[127, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_366, full_default_28);  unsqueeze_366 = full_default_28 = None
    full_default_31: "f32[127, 50400]" = torch.ops.aten.full.default([127, 50400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[127, 50400]" = torch.ops.aten.scatter.value(full_default_31, 1, where_30, -1.0);  full_default_31 = where_30 = None
    where_31: "f32[127, 1]" = torch.ops.aten.where.self(ne_3, div_57, full_default_29);  ne_3 = div_57 = None
    mul_282: "f32[127, 50400]" = torch.ops.aten.mul.Tensor(scatter, where_31);  scatter = where_31 = None
    exp_29: "f32[127, 50400]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_32: "f32[127, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [1], True)
    mul_283: "f32[127, 50400]" = torch.ops.aten.mul.Tensor(exp_29, sum_32);  exp_29 = sum_32 = None
    sub_59: "f32[127, 50400]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    view_791: "f32[1, 127, 50400]" = torch.ops.aten.reshape.default(sub_59, [1, 127, 50400]);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:885, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    full_default_33: "f32[1, 127, 50400]" = torch.ops.aten.full.default([1, 127, 50400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_34: "f32[1, 128, 50400]" = torch.ops.aten.full.default([1, 128, 50400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 128, 50400]" = torch.ops.aten.slice_scatter.default(full_default_34, view_791, 1, 0, -1);  full_default_34 = view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:885, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    add_226: "f32[1, 128, 50400]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:878, code: lm_logits = self.lm_head(hidden_states).to(torch.float32)
    view_792: "f32[128, 50400]" = torch.ops.aten.reshape.default(add_226, [128, 50400]);  add_226 = None
    mm_112: "f32[128, 4096]" = torch.ops.aten.mm.default(view_792, permute_309);  permute_309 = None
    permute_310: "f32[50400, 128]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_113: "f32[50400, 4096]" = torch.ops.aten.mm.default(permute_310, view_787);  permute_310 = view_787 = None
    permute_311: "f32[4096, 50400]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_33: "f32[1, 50400]" = torch.ops.aten.sum.dim_IntList(view_792, [0], True);  view_792 = None
    view_793: "f32[50400]" = torch.ops.aten.reshape.default(sum_33, [50400]);  sum_33 = None
    permute_312: "f32[50400, 4096]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_794: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_112, [1, 128, 4096]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    mul_285: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_794, primals_282);  primals_282 = None
    mul_286: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_285, 4096)
    sum_34: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
    mul_287: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_285, mul_280);  mul_285 = None
    sum_35: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
    mul_288: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, sum_35);  sum_35 = None
    sub_61: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_286, sum_34);  mul_286 = sum_34 = None
    sub_62: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_61, mul_288);  sub_61 = mul_288 = None
    mul_289: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_58, sub_62);  div_58 = sub_62 = None
    mul_290: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_794, mul_280);  mul_280 = None
    sum_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
    sum_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_794, [0, 1]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_796: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_289, [128, 4096])
    mm_114: "f32[128, 16384]" = torch.ops.aten.mm.default(view_796, permute_313);  permute_313 = None
    permute_314: "f32[4096, 128]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_115: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_314, view_784);  view_784 = None
    permute_315: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_38: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True)
    view_797: "f32[4096]" = torch.ops.aten.reshape.default(sum_38, [4096]);  sum_38 = None
    permute_316: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    view_798: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_114, [1, 128, 16384]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_291: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_798, mul_276);  mul_276 = None
    mul_292: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_798, add_221);  view_798 = add_221 = None
    mul_293: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
    sub_63: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_293);  mul_293 = None
    mul_294: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_291, sub_63);  mul_291 = sub_63 = None
    mul_295: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_294, 0.7978845608028654);  mul_294 = None
    mul_296: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_295, 0.044715)
    pow_29: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_783, 2.0);  view_783 = None
    mul_297: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
    mul_298: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_227: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_295, mul_298);  mul_295 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_299: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_292, 0.5);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_228: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_227, mul_299);  add_227 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_799: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_228, [128, 16384]);  add_228 = None
    mm_116: "f32[128, 4096]" = torch.ops.aten.mm.default(view_799, permute_317);  permute_317 = None
    permute_318: "f32[16384, 128]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_117: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_318, view_758);  permute_318 = None
    permute_319: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_39: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[16384]" = torch.ops.aten.reshape.default(sum_39, [16384]);  sum_39 = None
    permute_320: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_801: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_116, [1, 128, 4096]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_118: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_314, view_780);  permute_314 = view_780 = None
    permute_322: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    mm_119: "f32[128, 4096]" = torch.ops.aten.mm.default(view_796, permute_323);  view_796 = permute_323 = None
    view_803: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 4096]);  mm_119 = None
    permute_324: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_804: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_803, [1, 128, 16, 256]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_325: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_805: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_325, [16, 128, 256]);  permute_325 = None
    bmm_56: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_326, view_805);  permute_326 = None
    bmm_57: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_805, permute_327);  view_805 = permute_327 = None
    view_806: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 256]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_229: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_58, view_806);  tangents_58 = view_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_807: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_300: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_807, alias_59);  view_807 = None
    sum_40: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [-1], True)
    mul_301: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_59, sum_40);  alias_59 = sum_40 = None
    sub_64: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_59: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_64, primals_369);  sub_64 = primals_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_32: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1344, div_59, full_default_29);  slice_1344 = div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_808: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_32, [16, 128, 128]);  where_32 = None
    bmm_58: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_328, view_808);  permute_328 = None
    bmm_59: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_808, permute_329);  view_808 = permute_329 = None
    view_809: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_58, [1, 16, 256, 128]);  bmm_58 = None
    view_810: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_59, [1, 16, 128, 256]);  bmm_59 = None
    permute_330: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_809, [0, 1, 3, 2]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_230: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_57, permute_330);  tangents_57 = permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_331: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_332: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_230, [0, 2, 1, 3]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1348: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_331, 3, 0, 64)
    slice_1349: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_331, 3, 64, 256);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1350: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64)
    slice_1351: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 256);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_302: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1348, view_767)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_811: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_302, [1, 128, 16, 32, 2]);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_811, 4, 0)
    select_1: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_811, 4, 1);  view_811 = None
    neg_57: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    full_default_36: "f32[1, 128, 16, 64]" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_57, 3, 1, 9223372036854775807, 2);  neg_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_6: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_231: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_6);  slice_scatter_2 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_303: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1348, view_768);  slice_1348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_232: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_231, mul_303);  add_231 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_304: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1350, view_767);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_812: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_304, [1, 128, 16, 32, 2]);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_2: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_812, 4, 0)
    select_3: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_812, 4, 1);  view_812 = None
    neg_58: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_2);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_10: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_14: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_233: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_14);  slice_scatter_10 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_305: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1350, view_768);  slice_1350 = view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_234: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_233, mul_305);  add_233 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    full_default_52: "f32[1, 128, 16, 256]" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1349, 3, 64, 9223372036854775807);  slice_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_22: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_232, 3, 0, 64);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_235: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_22);  slice_scatter_18 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_26: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1351, 3, 64, 9223372036854775807);  slice_1351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_30: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_234, 3, 0, 64);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_236: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_30);  slice_scatter_26 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_333: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_229, [0, 2, 1, 3]);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_225: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    view_813: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_225, [1, 128, 4096]);  clone_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_814: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_236, [1, 128, 4096]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_815: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_235, [1, 128, 4096]);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_816: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_813, [128, 4096]);  view_813 = None
    permute_334: "f32[4096, 128]" = torch.ops.aten.permute.default(view_816, [1, 0])
    mm_120: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_334, view_758);  permute_334 = None
    permute_335: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    mm_121: "f32[128, 4096]" = torch.ops.aten.mm.default(view_816, permute_336);  view_816 = permute_336 = None
    view_817: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 4096]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_237: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_801, view_817);  view_801 = view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_337: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_818: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_814, [128, 4096]);  view_814 = None
    permute_338: "f32[4096, 128]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_122: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_338, view_758);  permute_338 = None
    permute_339: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    mm_123: "f32[128, 4096]" = torch.ops.aten.mm.default(view_818, permute_340);  view_818 = permute_340 = None
    view_819: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_238: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_237, view_819);  add_237 = view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_341: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_820: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_815, [128, 4096]);  view_815 = None
    permute_342: "f32[4096, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
    mm_124: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_342, view_758);  permute_342 = view_758 = None
    permute_343: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    mm_125: "f32[128, 4096]" = torch.ops.aten.mm.default(view_820, permute_344);  view_820 = permute_344 = None
    view_821: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 4096]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_239: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_238, view_821);  add_238 = view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_345: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_307: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_239, primals_272);  primals_272 = None
    mul_308: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_307, 4096)
    sum_41: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
    mul_309: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_307, mul_270);  mul_307 = None
    sum_42: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
    mul_310: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, sum_42);  sum_42 = None
    sub_66: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_308, sum_41);  mul_308 = sum_41 = None
    sub_67: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_66, mul_310);  sub_66 = mul_310 = None
    mul_311: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_60, sub_67);  div_60 = sub_67 = None
    mul_312: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_239, mul_270);  mul_270 = None
    sum_43: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
    sum_44: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_240: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_289, mul_311);  mul_289 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_822: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_240, [128, 4096])
    mm_126: "f32[128, 16384]" = torch.ops.aten.mm.default(view_822, permute_346);  permute_346 = None
    permute_347: "f32[4096, 128]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_127: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_347, view_756);  view_756 = None
    permute_348: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_45: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True)
    view_823: "f32[4096]" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
    permute_349: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_824: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_126, [1, 128, 16384]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_313: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_824, mul_266);  mul_266 = None
    mul_314: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_824, add_213);  view_824 = add_213 = None
    mul_315: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
    sub_68: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_315);  mul_315 = None
    mul_316: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_313, sub_68);  mul_313 = sub_68 = None
    mul_317: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_316, 0.7978845608028654);  mul_316 = None
    mul_318: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_317, 0.044715)
    pow_30: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_755, 2.0);  view_755 = None
    mul_319: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
    mul_320: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_241: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_317, mul_320);  mul_317 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_321: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_314, 0.5);  mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_242: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_241, mul_321);  add_241 = mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_825: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_242, [128, 16384]);  add_242 = None
    mm_128: "f32[128, 4096]" = torch.ops.aten.mm.default(view_825, permute_350);  permute_350 = None
    permute_351: "f32[16384, 128]" = torch.ops.aten.permute.default(view_825, [1, 0])
    mm_129: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_351, view_730);  permute_351 = None
    permute_352: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_46: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_825, [0], True);  view_825 = None
    view_826: "f32[16384]" = torch.ops.aten.reshape.default(sum_46, [16384]);  sum_46 = None
    permute_353: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_827: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_128, [1, 128, 4096]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_130: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_347, view_752);  permute_347 = view_752 = None
    permute_355: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[128, 4096]" = torch.ops.aten.mm.default(view_822, permute_356);  view_822 = permute_356 = None
    view_829: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 4096]);  mm_131 = None
    permute_357: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_830: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_829, [1, 128, 16, 256]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_358: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_831: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_358, [16, 128, 256]);  permute_358 = None
    bmm_60: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_359, view_831);  permute_359 = None
    bmm_61: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_831, permute_360);  view_831 = permute_360 = None
    view_832: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 256]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_243: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_56, view_832);  tangents_56 = view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_833: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_322: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_833, alias_61);  view_833 = None
    sum_47: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
    mul_323: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_61, sum_47);  alias_61 = sum_47 = None
    sub_69: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_61: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_69, primals_366);  sub_69 = primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_33: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1296, div_61, full_default_29);  slice_1296 = div_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_834: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_33, [16, 128, 128]);  where_33 = None
    bmm_62: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_361, view_834);  permute_361 = None
    bmm_63: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_834, permute_362);  view_834 = permute_362 = None
    view_835: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_62, [1, 16, 256, 128]);  bmm_62 = None
    view_836: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_63, [1, 16, 128, 256]);  bmm_63 = None
    permute_363: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_835, [0, 1, 3, 2]);  view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_244: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_55, permute_363);  tangents_55 = permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_364: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_365: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_244, [0, 2, 1, 3]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1352: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_364, 3, 0, 64)
    slice_1353: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_364, 3, 64, 256);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1354: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_365, 3, 0, 64)
    slice_1355: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_365, 3, 64, 256);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_324: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1352, view_739)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_837: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_324, [1, 128, 16, 32, 2]);  mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_4: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_837, 4, 0)
    select_5: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_837, 4, 1);  view_837 = None
    neg_59: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_4);  select_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_34: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_38: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_245: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_38);  slice_scatter_34 = slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_325: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1352, view_740);  slice_1352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_246: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_245, mul_325);  add_245 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_326: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1354, view_739);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_838: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_326, [1, 128, 16, 32, 2]);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_6: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_838, 4, 0)
    select_7: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_838, 4, 1);  view_838 = None
    neg_60: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_6);  select_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_42: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_60, 3, 1, 9223372036854775807, 2);  neg_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_46: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_247: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_46);  slice_scatter_42 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_327: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1354, view_740);  slice_1354 = view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_248: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_247, mul_327);  add_247 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_50: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1353, 3, 64, 9223372036854775807);  slice_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_54: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_246, 3, 0, 64);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_249: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_54);  slice_scatter_50 = slice_scatter_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_58: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1355, 3, 64, 9223372036854775807);  slice_1355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_62: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_248, 3, 0, 64);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_250: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_62);  slice_scatter_58 = slice_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_366: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_243, [0, 2, 1, 3]);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_226: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    view_839: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_226, [1, 128, 4096]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_840: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_250, [1, 128, 4096]);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_841: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_249, [1, 128, 4096]);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_842: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_839, [128, 4096]);  view_839 = None
    permute_367: "f32[4096, 128]" = torch.ops.aten.permute.default(view_842, [1, 0])
    mm_132: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_367, view_730);  permute_367 = None
    permute_368: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    mm_133: "f32[128, 4096]" = torch.ops.aten.mm.default(view_842, permute_369);  view_842 = permute_369 = None
    view_843: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 4096]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_251: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_827, view_843);  view_827 = view_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_370: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_844: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_840, [128, 4096]);  view_840 = None
    permute_371: "f32[4096, 128]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_134: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_371, view_730);  permute_371 = None
    permute_372: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    mm_135: "f32[128, 4096]" = torch.ops.aten.mm.default(view_844, permute_373);  view_844 = permute_373 = None
    view_845: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 4096]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_252: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_251, view_845);  add_251 = view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_374: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_846: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_841, [128, 4096]);  view_841 = None
    permute_375: "f32[4096, 128]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_136: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_375, view_730);  permute_375 = view_730 = None
    permute_376: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[128, 4096]" = torch.ops.aten.mm.default(view_846, permute_377);  view_846 = permute_377 = None
    view_847: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 4096]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_253: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_252, view_847);  add_252 = view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_378: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_329: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_253, primals_262);  primals_262 = None
    mul_330: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_329, 4096)
    sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_329, mul_260);  mul_329 = None
    sum_49: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, sum_49);  sum_49 = None
    sub_71: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_330, sum_48);  mul_330 = sum_48 = None
    sub_72: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_71, mul_332);  sub_71 = mul_332 = None
    mul_333: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_62, sub_72);  div_62 = sub_72 = None
    mul_334: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_253, mul_260);  mul_260 = None
    sum_50: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_51: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_253, [0, 1]);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_254: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_240, mul_333);  add_240 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_848: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_254, [128, 4096])
    mm_138: "f32[128, 16384]" = torch.ops.aten.mm.default(view_848, permute_379);  permute_379 = None
    permute_380: "f32[4096, 128]" = torch.ops.aten.permute.default(view_848, [1, 0])
    mm_139: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_380, view_728);  view_728 = None
    permute_381: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True)
    view_849: "f32[4096]" = torch.ops.aten.reshape.default(sum_52, [4096]);  sum_52 = None
    permute_382: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_850: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_138, [1, 128, 16384]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_335: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_850, mul_256);  mul_256 = None
    mul_336: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_850, add_205);  view_850 = add_205 = None
    mul_337: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
    sub_73: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
    mul_338: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_335, sub_73);  mul_335 = sub_73 = None
    mul_339: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
    mul_340: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_339, 0.044715)
    pow_31: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_727, 2.0);  view_727 = None
    mul_341: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
    mul_342: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_255: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_339, mul_342);  mul_339 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_343: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_336, 0.5);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_256: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_255, mul_343);  add_255 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_851: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_256, [128, 16384]);  add_256 = None
    mm_140: "f32[128, 4096]" = torch.ops.aten.mm.default(view_851, permute_383);  permute_383 = None
    permute_384: "f32[16384, 128]" = torch.ops.aten.permute.default(view_851, [1, 0])
    mm_141: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_384, view_702);  permute_384 = None
    permute_385: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_53: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
    view_852: "f32[16384]" = torch.ops.aten.reshape.default(sum_53, [16384]);  sum_53 = None
    permute_386: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_853: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_140, [1, 128, 4096]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_142: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_380, view_724);  permute_380 = view_724 = None
    permute_388: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    mm_143: "f32[128, 4096]" = torch.ops.aten.mm.default(view_848, permute_389);  view_848 = permute_389 = None
    view_855: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None
    permute_390: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_856: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_855, [1, 128, 16, 256]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_391: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_857: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_391, [16, 128, 256]);  permute_391 = None
    bmm_64: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_392, view_857);  permute_392 = None
    bmm_65: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_857, permute_393);  view_857 = permute_393 = None
    view_858: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 256]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_257: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_54, view_858);  tangents_54 = view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_859: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_344: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_859, alias_63);  view_859 = None
    sum_54: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_63, sum_54);  alias_63 = sum_54 = None
    sub_74: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_63: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_74, primals_363);  sub_74 = primals_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_34: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1248, div_63, full_default_29);  slice_1248 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_860: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_34, [16, 128, 128]);  where_34 = None
    bmm_66: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_394, view_860);  permute_394 = None
    bmm_67: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_860, permute_395);  view_860 = permute_395 = None
    view_861: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_66, [1, 16, 256, 128]);  bmm_66 = None
    view_862: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_67, [1, 16, 128, 256]);  bmm_67 = None
    permute_396: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_861, [0, 1, 3, 2]);  view_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_258: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_53, permute_396);  tangents_53 = permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_397: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_398: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_258, [0, 2, 1, 3]);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1356: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_397, 3, 0, 64)
    slice_1357: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_397, 3, 64, 256);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1358: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_398, 3, 0, 64)
    slice_1359: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_398, 3, 64, 256);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_346: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1356, view_711)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_863: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_346, [1, 128, 16, 32, 2]);  mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_8: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_863, 4, 0)
    select_9: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_863, 4, 1);  view_863 = None
    neg_61: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_8);  select_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_66: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_70: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_259: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_70);  slice_scatter_66 = slice_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_347: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1356, view_712);  slice_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_260: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_259, mul_347);  add_259 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_348: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1358, view_711);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_864: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_348, [1, 128, 16, 32, 2]);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_10: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_864, 4, 0)
    select_11: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_864, 4, 1);  view_864 = None
    neg_62: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_10);  select_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_74: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_78: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_261: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_78);  slice_scatter_74 = slice_scatter_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_349: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1358, view_712);  slice_1358 = view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_262: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_261, mul_349);  add_261 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_82: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1357, 3, 64, 9223372036854775807);  slice_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_86: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_260, 3, 0, 64);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_263: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_86);  slice_scatter_82 = slice_scatter_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_90: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1359, 3, 64, 9223372036854775807);  slice_1359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_94: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_262, 3, 0, 64);  add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_264: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_94);  slice_scatter_90 = slice_scatter_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_399: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_257, [0, 2, 1, 3]);  add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_227: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
    view_865: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_227, [1, 128, 4096]);  clone_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_866: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_264, [1, 128, 4096]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_867: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_263, [1, 128, 4096]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_868: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_865, [128, 4096]);  view_865 = None
    permute_400: "f32[4096, 128]" = torch.ops.aten.permute.default(view_868, [1, 0])
    mm_144: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_400, view_702);  permute_400 = None
    permute_401: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[128, 4096]" = torch.ops.aten.mm.default(view_868, permute_402);  view_868 = permute_402 = None
    view_869: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_145, [1, 128, 4096]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_265: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_853, view_869);  view_853 = view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_403: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_870: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_866, [128, 4096]);  view_866 = None
    permute_404: "f32[4096, 128]" = torch.ops.aten.permute.default(view_870, [1, 0])
    mm_146: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_404, view_702);  permute_404 = None
    permute_405: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    mm_147: "f32[128, 4096]" = torch.ops.aten.mm.default(view_870, permute_406);  view_870 = permute_406 = None
    view_871: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_147, [1, 128, 4096]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_266: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_265, view_871);  add_265 = view_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_407: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_872: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_867, [128, 4096]);  view_867 = None
    permute_408: "f32[4096, 128]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_148: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_408, view_702);  permute_408 = view_702 = None
    permute_409: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    mm_149: "f32[128, 4096]" = torch.ops.aten.mm.default(view_872, permute_410);  view_872 = permute_410 = None
    view_873: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_149, [1, 128, 4096]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_267: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_266, view_873);  add_266 = view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_411: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_351: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_267, primals_252);  primals_252 = None
    mul_352: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_351, 4096)
    sum_55: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True)
    mul_353: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_351, mul_250);  mul_351 = None
    sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
    mul_354: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, sum_56);  sum_56 = None
    sub_76: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_352, sum_55);  mul_352 = sum_55 = None
    sub_77: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_76, mul_354);  sub_76 = mul_354 = None
    mul_355: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_64, sub_77);  div_64 = sub_77 = None
    mul_356: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_267, mul_250);  mul_250 = None
    sum_57: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1]);  mul_356 = None
    sum_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_268: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_254, mul_355);  add_254 = mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_874: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_268, [128, 4096])
    mm_150: "f32[128, 16384]" = torch.ops.aten.mm.default(view_874, permute_412);  permute_412 = None
    permute_413: "f32[4096, 128]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_151: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_413, view_700);  view_700 = None
    permute_414: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_59: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True)
    view_875: "f32[4096]" = torch.ops.aten.reshape.default(sum_59, [4096]);  sum_59 = None
    permute_415: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    view_876: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_150, [1, 128, 16384]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_357: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_876, mul_246);  mul_246 = None
    mul_358: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_876, add_197);  view_876 = add_197 = None
    mul_359: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
    sub_78: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_359);  mul_359 = None
    mul_360: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_357, sub_78);  mul_357 = sub_78 = None
    mul_361: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_360, 0.7978845608028654);  mul_360 = None
    mul_362: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_361, 0.044715)
    pow_32: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_699, 2.0);  view_699 = None
    mul_363: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
    mul_364: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_269: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_361, mul_364);  mul_361 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_365: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_358, 0.5);  mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_270: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_269, mul_365);  add_269 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_877: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_270, [128, 16384]);  add_270 = None
    mm_152: "f32[128, 4096]" = torch.ops.aten.mm.default(view_877, permute_416);  permute_416 = None
    permute_417: "f32[16384, 128]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_153: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_417, view_674);  permute_417 = None
    permute_418: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_60: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[16384]" = torch.ops.aten.reshape.default(sum_60, [16384]);  sum_60 = None
    permute_419: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    view_879: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_152, [1, 128, 4096]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_154: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_413, view_696);  permute_413 = view_696 = None
    permute_421: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    mm_155: "f32[128, 4096]" = torch.ops.aten.mm.default(view_874, permute_422);  view_874 = permute_422 = None
    view_881: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_155, [1, 128, 4096]);  mm_155 = None
    permute_423: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_882: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_881, [1, 128, 16, 256]);  view_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_424: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_883: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_424, [16, 128, 256]);  permute_424 = None
    bmm_68: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_425, view_883);  permute_425 = None
    bmm_69: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_883, permute_426);  view_883 = permute_426 = None
    view_884: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 256]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_271: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_52, view_884);  tangents_52 = view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_885: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_366: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, alias_65);  view_885 = None
    sum_61: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [-1], True)
    mul_367: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_65, sum_61);  alias_65 = sum_61 = None
    sub_79: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_65: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_79, primals_360);  sub_79 = primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_35: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1200, div_65, full_default_29);  slice_1200 = div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_886: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_35, [16, 128, 128]);  where_35 = None
    bmm_70: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_427, view_886);  permute_427 = None
    bmm_71: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_886, permute_428);  view_886 = permute_428 = None
    view_887: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_70, [1, 16, 256, 128]);  bmm_70 = None
    view_888: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_71, [1, 16, 128, 256]);  bmm_71 = None
    permute_429: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_887, [0, 1, 3, 2]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_272: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_51, permute_429);  tangents_51 = permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_430: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_431: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_272, [0, 2, 1, 3]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1360: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_430, 3, 0, 64)
    slice_1361: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_430, 3, 64, 256);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1362: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_431, 3, 0, 64)
    slice_1363: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_431, 3, 64, 256);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_368: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1360, view_683)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_889: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_368, [1, 128, 16, 32, 2]);  mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_12: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_889, 4, 0)
    select_13: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_889, 4, 1);  view_889 = None
    neg_63: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_12);  select_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_98: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_63, 3, 1, 9223372036854775807, 2);  neg_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_102: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_273: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_102);  slice_scatter_98 = slice_scatter_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_369: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1360, view_684);  slice_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_274: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_273, mul_369);  add_273 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_370: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1362, view_683);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_890: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_370, [1, 128, 16, 32, 2]);  mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_14: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_890, 4, 0)
    select_15: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_890, 4, 1);  view_890 = None
    neg_64: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_14);  select_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_106: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_110: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_275: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_110);  slice_scatter_106 = slice_scatter_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_371: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1362, view_684);  slice_1362 = view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_276: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_275, mul_371);  add_275 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_114: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1361, 3, 64, 9223372036854775807);  slice_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_118: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_274, 3, 0, 64);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_277: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_118);  slice_scatter_114 = slice_scatter_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_122: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1363, 3, 64, 9223372036854775807);  slice_1363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_126: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_276, 3, 0, 64);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_278: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_126);  slice_scatter_122 = slice_scatter_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_432: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_271, [0, 2, 1, 3]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_228: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
    view_891: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_228, [1, 128, 4096]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_892: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_278, [1, 128, 4096]);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_893: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_277, [1, 128, 4096]);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_894: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_891, [128, 4096]);  view_891 = None
    permute_433: "f32[4096, 128]" = torch.ops.aten.permute.default(view_894, [1, 0])
    mm_156: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_433, view_674);  permute_433 = None
    permute_434: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    mm_157: "f32[128, 4096]" = torch.ops.aten.mm.default(view_894, permute_435);  view_894 = permute_435 = None
    view_895: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_157, [1, 128, 4096]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_279: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_879, view_895);  view_879 = view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_436: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_896: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_892, [128, 4096]);  view_892 = None
    permute_437: "f32[4096, 128]" = torch.ops.aten.permute.default(view_896, [1, 0])
    mm_158: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_437, view_674);  permute_437 = None
    permute_438: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    mm_159: "f32[128, 4096]" = torch.ops.aten.mm.default(view_896, permute_439);  view_896 = permute_439 = None
    view_897: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_159, [1, 128, 4096]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_280: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_279, view_897);  add_279 = view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_440: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_898: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_893, [128, 4096]);  view_893 = None
    permute_441: "f32[4096, 128]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_160: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_441, view_674);  permute_441 = view_674 = None
    permute_442: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    mm_161: "f32[128, 4096]" = torch.ops.aten.mm.default(view_898, permute_443);  view_898 = permute_443 = None
    view_899: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_161, [1, 128, 4096]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_281: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_280, view_899);  add_280 = view_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_444: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_373: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_281, primals_242);  primals_242 = None
    mul_374: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_373, 4096)
    sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True)
    mul_375: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_373, mul_240);  mul_373 = None
    sum_63: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    mul_376: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, sum_63);  sum_63 = None
    sub_81: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_374, sum_62);  mul_374 = sum_62 = None
    sub_82: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_81, mul_376);  sub_81 = mul_376 = None
    mul_377: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_66, sub_82);  div_66 = sub_82 = None
    mul_378: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_281, mul_240);  mul_240 = None
    sum_64: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1]);  mul_378 = None
    sum_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_282: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_268, mul_377);  add_268 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_900: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_282, [128, 4096])
    mm_162: "f32[128, 16384]" = torch.ops.aten.mm.default(view_900, permute_445);  permute_445 = None
    permute_446: "f32[4096, 128]" = torch.ops.aten.permute.default(view_900, [1, 0])
    mm_163: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_446, view_672);  view_672 = None
    permute_447: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True)
    view_901: "f32[4096]" = torch.ops.aten.reshape.default(sum_66, [4096]);  sum_66 = None
    permute_448: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    view_902: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_162, [1, 128, 16384]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_379: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_902, mul_236);  mul_236 = None
    mul_380: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_902, add_189);  view_902 = add_189 = None
    mul_381: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
    sub_83: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_381);  mul_381 = None
    mul_382: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_379, sub_83);  mul_379 = sub_83 = None
    mul_383: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_382, 0.7978845608028654);  mul_382 = None
    mul_384: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_383, 0.044715)
    pow_33: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_671, 2.0);  view_671 = None
    mul_385: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
    mul_386: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_283: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_383, mul_386);  mul_383 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_387: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_380, 0.5);  mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_284: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_283, mul_387);  add_283 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_903: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_284, [128, 16384]);  add_284 = None
    mm_164: "f32[128, 4096]" = torch.ops.aten.mm.default(view_903, permute_449);  permute_449 = None
    permute_450: "f32[16384, 128]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_165: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_450, view_646);  permute_450 = None
    permute_451: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_67: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[16384]" = torch.ops.aten.reshape.default(sum_67, [16384]);  sum_67 = None
    permute_452: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
    view_905: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_164, [1, 128, 4096]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_166: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_446, view_668);  permute_446 = view_668 = None
    permute_454: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[128, 4096]" = torch.ops.aten.mm.default(view_900, permute_455);  view_900 = permute_455 = None
    view_907: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_167, [1, 128, 4096]);  mm_167 = None
    permute_456: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_908: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_907, [1, 128, 16, 256]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_457: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_908, [0, 2, 1, 3]);  view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_909: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_457, [16, 128, 256]);  permute_457 = None
    bmm_72: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_458, view_909);  permute_458 = None
    bmm_73: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_909, permute_459);  view_909 = permute_459 = None
    view_910: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_72, [1, 16, 128, 256]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_285: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_50, view_910);  tangents_50 = view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_911: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_388: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_911, alias_67);  view_911 = None
    sum_68: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [-1], True)
    mul_389: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_67, sum_68);  alias_67 = sum_68 = None
    sub_84: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_67: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_84, primals_357);  sub_84 = primals_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_36: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1152, div_67, full_default_29);  slice_1152 = div_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_912: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_36, [16, 128, 128]);  where_36 = None
    bmm_74: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_460, view_912);  permute_460 = None
    bmm_75: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_912, permute_461);  view_912 = permute_461 = None
    view_913: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_74, [1, 16, 256, 128]);  bmm_74 = None
    view_914: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_75, [1, 16, 128, 256]);  bmm_75 = None
    permute_462: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_913, [0, 1, 3, 2]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_286: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_49, permute_462);  tangents_49 = permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_463: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_464: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_286, [0, 2, 1, 3]);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1364: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_463, 3, 0, 64)
    slice_1365: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_463, 3, 64, 256);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1366: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_464, 3, 0, 64)
    slice_1367: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_464, 3, 64, 256);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_390: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1364, view_655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_915: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_390, [1, 128, 16, 32, 2]);  mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_16: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_915, 4, 0)
    select_17: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_915, 4, 1);  view_915 = None
    neg_65: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_16);  select_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_130: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_134: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_287: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_134);  slice_scatter_130 = slice_scatter_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_391: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1364, view_656);  slice_1364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_288: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_287, mul_391);  add_287 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_392: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1366, view_655);  view_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_916: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_392, [1, 128, 16, 32, 2]);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_18: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_916, 4, 0)
    select_19: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_916, 4, 1);  view_916 = None
    neg_66: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_18);  select_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_138: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_66, 3, 1, 9223372036854775807, 2);  neg_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_142: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_289: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_142);  slice_scatter_138 = slice_scatter_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_393: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1366, view_656);  slice_1366 = view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_290: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_289, mul_393);  add_289 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_146: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1365, 3, 64, 9223372036854775807);  slice_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_150: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_288, 3, 0, 64);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_291: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_150);  slice_scatter_146 = slice_scatter_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_154: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1367, 3, 64, 9223372036854775807);  slice_1367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_158: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_290, 3, 0, 64);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_292: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_158);  slice_scatter_154 = slice_scatter_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_465: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_285, [0, 2, 1, 3]);  add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_229: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
    view_917: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_229, [1, 128, 4096]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_918: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_292, [1, 128, 4096]);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_919: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_291, [1, 128, 4096]);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_920: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_917, [128, 4096]);  view_917 = None
    permute_466: "f32[4096, 128]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_168: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_466, view_646);  permute_466 = None
    permute_467: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[128, 4096]" = torch.ops.aten.mm.default(view_920, permute_468);  view_920 = permute_468 = None
    view_921: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_169, [1, 128, 4096]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_293: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_905, view_921);  view_905 = view_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_469: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_922: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_918, [128, 4096]);  view_918 = None
    permute_470: "f32[4096, 128]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_170: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_470, view_646);  permute_470 = None
    permute_471: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    mm_171: "f32[128, 4096]" = torch.ops.aten.mm.default(view_922, permute_472);  view_922 = permute_472 = None
    view_923: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_171, [1, 128, 4096]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_294: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_293, view_923);  add_293 = view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_473: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_924: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_919, [128, 4096]);  view_919 = None
    permute_474: "f32[4096, 128]" = torch.ops.aten.permute.default(view_924, [1, 0])
    mm_172: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_474, view_646);  permute_474 = view_646 = None
    permute_475: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    mm_173: "f32[128, 4096]" = torch.ops.aten.mm.default(view_924, permute_476);  view_924 = permute_476 = None
    view_925: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_173, [1, 128, 4096]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_295: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_294, view_925);  add_294 = view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_477: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_395: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_295, primals_232);  primals_232 = None
    mul_396: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_395, 4096)
    sum_69: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_395, mul_230);  mul_395 = None
    sum_70: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, sum_70);  sum_70 = None
    sub_86: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_396, sum_69);  mul_396 = sum_69 = None
    sub_87: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_86, mul_398);  sub_86 = mul_398 = None
    mul_399: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_68, sub_87);  div_68 = sub_87 = None
    mul_400: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_295, mul_230);  mul_230 = None
    sum_71: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_72: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_295, [0, 1]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_296: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_282, mul_399);  add_282 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_926: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_296, [128, 4096])
    mm_174: "f32[128, 16384]" = torch.ops.aten.mm.default(view_926, permute_478);  permute_478 = None
    permute_479: "f32[4096, 128]" = torch.ops.aten.permute.default(view_926, [1, 0])
    mm_175: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_479, view_644);  view_644 = None
    permute_480: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_73: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True)
    view_927: "f32[4096]" = torch.ops.aten.reshape.default(sum_73, [4096]);  sum_73 = None
    permute_481: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    view_928: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_174, [1, 128, 16384]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_401: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_928, mul_226);  mul_226 = None
    mul_402: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_928, add_181);  view_928 = add_181 = None
    mul_403: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
    sub_88: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_403);  mul_403 = None
    mul_404: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_401, sub_88);  mul_401 = sub_88 = None
    mul_405: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_404, 0.7978845608028654);  mul_404 = None
    mul_406: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_405, 0.044715)
    pow_34: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_643, 2.0);  view_643 = None
    mul_407: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
    mul_408: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_297: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_405, mul_408);  mul_405 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_409: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_402, 0.5);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_298: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_297, mul_409);  add_297 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_929: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_298, [128, 16384]);  add_298 = None
    mm_176: "f32[128, 4096]" = torch.ops.aten.mm.default(view_929, permute_482);  permute_482 = None
    permute_483: "f32[16384, 128]" = torch.ops.aten.permute.default(view_929, [1, 0])
    mm_177: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_483, view_618);  permute_483 = None
    permute_484: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_74: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_929, [0], True);  view_929 = None
    view_930: "f32[16384]" = torch.ops.aten.reshape.default(sum_74, [16384]);  sum_74 = None
    permute_485: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    view_931: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_176, [1, 128, 4096]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_178: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_479, view_640);  permute_479 = view_640 = None
    permute_487: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    mm_179: "f32[128, 4096]" = torch.ops.aten.mm.default(view_926, permute_488);  view_926 = permute_488 = None
    view_933: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_179, [1, 128, 4096]);  mm_179 = None
    permute_489: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_934: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_933, [1, 128, 16, 256]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_490: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_935: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_490, [16, 128, 256]);  permute_490 = None
    bmm_76: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_491, view_935);  permute_491 = None
    bmm_77: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_935, permute_492);  view_935 = permute_492 = None
    view_936: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_76, [1, 16, 128, 256]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_299: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_48, view_936);  tangents_48 = view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_937: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_410: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_937, alias_69);  view_937 = None
    sum_75: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [-1], True)
    mul_411: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_69, sum_75);  alias_69 = sum_75 = None
    sub_89: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_69: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_89, primals_354);  sub_89 = primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_37: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1104, div_69, full_default_29);  slice_1104 = div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_938: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_37, [16, 128, 128]);  where_37 = None
    bmm_78: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_493, view_938);  permute_493 = None
    bmm_79: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_938, permute_494);  view_938 = permute_494 = None
    view_939: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_78, [1, 16, 256, 128]);  bmm_78 = None
    view_940: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_79, [1, 16, 128, 256]);  bmm_79 = None
    permute_495: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_939, [0, 1, 3, 2]);  view_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_300: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_47, permute_495);  tangents_47 = permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_496: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_940, [0, 2, 1, 3]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_497: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_300, [0, 2, 1, 3]);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1368: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_496, 3, 0, 64)
    slice_1369: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_496, 3, 64, 256);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1370: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_497, 3, 0, 64)
    slice_1371: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_497, 3, 64, 256);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_412: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1368, view_627)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_941: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_412, [1, 128, 16, 32, 2]);  mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_20: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_941, 4, 0)
    select_21: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_941, 4, 1);  view_941 = None
    neg_67: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_20);  select_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_162: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_166: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_301: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_166);  slice_scatter_162 = slice_scatter_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_413: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1368, view_628);  slice_1368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_302: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_301, mul_413);  add_301 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_414: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1370, view_627);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_942: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_414, [1, 128, 16, 32, 2]);  mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_22: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_942, 4, 0)
    select_23: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_942, 4, 1);  view_942 = None
    neg_68: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_22);  select_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_170: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_174: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_303: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_174);  slice_scatter_170 = slice_scatter_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_415: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1370, view_628);  slice_1370 = view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_304: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_303, mul_415);  add_303 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_178: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1369, 3, 64, 9223372036854775807);  slice_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_182: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_302, 3, 0, 64);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_305: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_182);  slice_scatter_178 = slice_scatter_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_186: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1371, 3, 64, 9223372036854775807);  slice_1371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_190: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_304, 3, 0, 64);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_306: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_190);  slice_scatter_186 = slice_scatter_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_498: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_299, [0, 2, 1, 3]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_230: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
    view_943: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_230, [1, 128, 4096]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_944: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_306, [1, 128, 4096]);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_945: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_305, [1, 128, 4096]);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_946: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_943, [128, 4096]);  view_943 = None
    permute_499: "f32[4096, 128]" = torch.ops.aten.permute.default(view_946, [1, 0])
    mm_180: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_499, view_618);  permute_499 = None
    permute_500: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    mm_181: "f32[128, 4096]" = torch.ops.aten.mm.default(view_946, permute_501);  view_946 = permute_501 = None
    view_947: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_181, [1, 128, 4096]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_307: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_931, view_947);  view_931 = view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_502: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_948: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_944, [128, 4096]);  view_944 = None
    permute_503: "f32[4096, 128]" = torch.ops.aten.permute.default(view_948, [1, 0])
    mm_182: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_503, view_618);  permute_503 = None
    permute_504: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    mm_183: "f32[128, 4096]" = torch.ops.aten.mm.default(view_948, permute_505);  view_948 = permute_505 = None
    view_949: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_308: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_307, view_949);  add_307 = view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_506: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_950: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_945, [128, 4096]);  view_945 = None
    permute_507: "f32[4096, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_184: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_507, view_618);  permute_507 = view_618 = None
    permute_508: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    mm_185: "f32[128, 4096]" = torch.ops.aten.mm.default(view_950, permute_509);  view_950 = permute_509 = None
    view_951: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_185, [1, 128, 4096]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_309: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_308, view_951);  add_308 = view_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_510: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_417: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_309, primals_222);  primals_222 = None
    mul_418: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_417, 4096)
    sum_76: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True)
    mul_419: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_417, mul_220);  mul_417 = None
    sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True);  mul_419 = None
    mul_420: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, sum_77);  sum_77 = None
    sub_91: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_418, sum_76);  mul_418 = sum_76 = None
    sub_92: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_91, mul_420);  sub_91 = mul_420 = None
    mul_421: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_70, sub_92);  div_70 = sub_92 = None
    mul_422: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_309, mul_220);  mul_220 = None
    sum_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 1]);  mul_422 = None
    sum_79: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_309, [0, 1]);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_310: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_296, mul_421);  add_296 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_952: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_310, [128, 4096])
    mm_186: "f32[128, 16384]" = torch.ops.aten.mm.default(view_952, permute_511);  permute_511 = None
    permute_512: "f32[4096, 128]" = torch.ops.aten.permute.default(view_952, [1, 0])
    mm_187: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_512, view_616);  view_616 = None
    permute_513: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True)
    view_953: "f32[4096]" = torch.ops.aten.reshape.default(sum_80, [4096]);  sum_80 = None
    permute_514: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    view_954: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_186, [1, 128, 16384]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_423: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_954, mul_216);  mul_216 = None
    mul_424: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_954, add_173);  view_954 = add_173 = None
    mul_425: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
    sub_93: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_425);  mul_425 = None
    mul_426: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_423, sub_93);  mul_423 = sub_93 = None
    mul_427: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_426, 0.7978845608028654);  mul_426 = None
    mul_428: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_427, 0.044715)
    pow_35: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_615, 2.0);  view_615 = None
    mul_429: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
    mul_430: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_311: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_427, mul_430);  mul_427 = mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_431: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_424, 0.5);  mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_312: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_311, mul_431);  add_311 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_955: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_312, [128, 16384]);  add_312 = None
    mm_188: "f32[128, 4096]" = torch.ops.aten.mm.default(view_955, permute_515);  permute_515 = None
    permute_516: "f32[16384, 128]" = torch.ops.aten.permute.default(view_955, [1, 0])
    mm_189: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_516, view_590);  permute_516 = None
    permute_517: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_81: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_955, [0], True);  view_955 = None
    view_956: "f32[16384]" = torch.ops.aten.reshape.default(sum_81, [16384]);  sum_81 = None
    permute_518: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    view_957: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_188, [1, 128, 4096]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_190: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_512, view_612);  permute_512 = view_612 = None
    permute_520: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    mm_191: "f32[128, 4096]" = torch.ops.aten.mm.default(view_952, permute_521);  view_952 = permute_521 = None
    view_959: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_191, [1, 128, 4096]);  mm_191 = None
    permute_522: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_960: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_959, [1, 128, 16, 256]);  view_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_523: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_961: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_523, [16, 128, 256]);  permute_523 = None
    bmm_80: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_524, view_961);  permute_524 = None
    bmm_81: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_961, permute_525);  view_961 = permute_525 = None
    view_962: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_80, [1, 16, 128, 256]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_313: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_46, view_962);  tangents_46 = view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_963: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_432: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_963, alias_71);  view_963 = None
    sum_82: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [-1], True)
    mul_433: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_71, sum_82);  alias_71 = sum_82 = None
    sub_94: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_71: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_94, primals_351);  sub_94 = primals_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_38: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1056, div_71, full_default_29);  slice_1056 = div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_964: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_38, [16, 128, 128]);  where_38 = None
    bmm_82: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_526, view_964);  permute_526 = None
    bmm_83: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_964, permute_527);  view_964 = permute_527 = None
    view_965: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_82, [1, 16, 256, 128]);  bmm_82 = None
    view_966: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_83, [1, 16, 128, 256]);  bmm_83 = None
    permute_528: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_965, [0, 1, 3, 2]);  view_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_314: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_45, permute_528);  tangents_45 = permute_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_529: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_966, [0, 2, 1, 3]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_530: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_314, [0, 2, 1, 3]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1372: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_529, 3, 0, 64)
    slice_1373: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_529, 3, 64, 256);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1374: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_530, 3, 0, 64)
    slice_1375: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_530, 3, 64, 256);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_434: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1372, view_599)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_967: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_434, [1, 128, 16, 32, 2]);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_24: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_967, 4, 0)
    select_25: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_967, 4, 1);  view_967 = None
    neg_69: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_24);  select_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_194: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_69, 3, 1, 9223372036854775807, 2);  neg_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_198: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_315: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_198);  slice_scatter_194 = slice_scatter_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_435: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1372, view_600);  slice_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_316: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_315, mul_435);  add_315 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_436: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1374, view_599);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_968: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_436, [1, 128, 16, 32, 2]);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_26: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_968, 4, 0)
    select_27: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_968, 4, 1);  view_968 = None
    neg_70: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_26);  select_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_202: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_206: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_317: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_206);  slice_scatter_202 = slice_scatter_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_437: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1374, view_600);  slice_1374 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_318: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_317, mul_437);  add_317 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_210: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1373, 3, 64, 9223372036854775807);  slice_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_214: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_316, 3, 0, 64);  add_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_319: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_214);  slice_scatter_210 = slice_scatter_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_218: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1375, 3, 64, 9223372036854775807);  slice_1375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_222: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_318, 3, 0, 64);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_320: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_222);  slice_scatter_218 = slice_scatter_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_531: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_313, [0, 2, 1, 3]);  add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_231: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
    view_969: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_231, [1, 128, 4096]);  clone_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_970: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_320, [1, 128, 4096]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_971: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_319, [1, 128, 4096]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_972: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_969, [128, 4096]);  view_969 = None
    permute_532: "f32[4096, 128]" = torch.ops.aten.permute.default(view_972, [1, 0])
    mm_192: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_532, view_590);  permute_532 = None
    permute_533: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    mm_193: "f32[128, 4096]" = torch.ops.aten.mm.default(view_972, permute_534);  view_972 = permute_534 = None
    view_973: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_193, [1, 128, 4096]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_321: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_957, view_973);  view_957 = view_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_535: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_974: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_970, [128, 4096]);  view_970 = None
    permute_536: "f32[4096, 128]" = torch.ops.aten.permute.default(view_974, [1, 0])
    mm_194: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_536, view_590);  permute_536 = None
    permute_537: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    mm_195: "f32[128, 4096]" = torch.ops.aten.mm.default(view_974, permute_538);  view_974 = permute_538 = None
    view_975: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_195, [1, 128, 4096]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_322: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_321, view_975);  add_321 = view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_539: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_976: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_971, [128, 4096]);  view_971 = None
    permute_540: "f32[4096, 128]" = torch.ops.aten.permute.default(view_976, [1, 0])
    mm_196: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_540, view_590);  permute_540 = view_590 = None
    permute_541: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    mm_197: "f32[128, 4096]" = torch.ops.aten.mm.default(view_976, permute_542);  view_976 = permute_542 = None
    view_977: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_197, [1, 128, 4096]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_323: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_322, view_977);  add_322 = view_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_543: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_439: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_323, primals_212);  primals_212 = None
    mul_440: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_439, 4096)
    sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
    mul_441: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_439, mul_210);  mul_439 = None
    sum_84: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
    mul_442: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, sum_84);  sum_84 = None
    sub_96: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_440, sum_83);  mul_440 = sum_83 = None
    sub_97: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_96, mul_442);  sub_96 = mul_442 = None
    mul_443: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_72, sub_97);  div_72 = sub_97 = None
    mul_444: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_323, mul_210);  mul_210 = None
    sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
    sum_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_324: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_310, mul_443);  add_310 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_978: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_324, [128, 4096])
    mm_198: "f32[128, 16384]" = torch.ops.aten.mm.default(view_978, permute_544);  permute_544 = None
    permute_545: "f32[4096, 128]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_199: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_545, view_588);  view_588 = None
    permute_546: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True)
    view_979: "f32[4096]" = torch.ops.aten.reshape.default(sum_87, [4096]);  sum_87 = None
    permute_547: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_980: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_198, [1, 128, 16384]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_445: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_980, mul_206);  mul_206 = None
    mul_446: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_980, add_165);  view_980 = add_165 = None
    mul_447: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
    sub_98: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_447);  mul_447 = None
    mul_448: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_445, sub_98);  mul_445 = sub_98 = None
    mul_449: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_448, 0.7978845608028654);  mul_448 = None
    mul_450: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_449, 0.044715)
    pow_36: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_587, 2.0);  view_587 = None
    mul_451: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
    mul_452: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_325: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_449, mul_452);  mul_449 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_453: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_446, 0.5);  mul_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_326: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_325, mul_453);  add_325 = mul_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_981: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_326, [128, 16384]);  add_326 = None
    mm_200: "f32[128, 4096]" = torch.ops.aten.mm.default(view_981, permute_548);  permute_548 = None
    permute_549: "f32[16384, 128]" = torch.ops.aten.permute.default(view_981, [1, 0])
    mm_201: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_549, view_562);  permute_549 = None
    permute_550: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_981, [0], True);  view_981 = None
    view_982: "f32[16384]" = torch.ops.aten.reshape.default(sum_88, [16384]);  sum_88 = None
    permute_551: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_983: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_200, [1, 128, 4096]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_202: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_545, view_584);  permute_545 = view_584 = None
    permute_553: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    mm_203: "f32[128, 4096]" = torch.ops.aten.mm.default(view_978, permute_554);  view_978 = permute_554 = None
    view_985: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None
    permute_555: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_986: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_985, [1, 128, 16, 256]);  view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_556: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_986, [0, 2, 1, 3]);  view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_987: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_556, [16, 128, 256]);  permute_556 = None
    bmm_84: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_557, view_987);  permute_557 = None
    bmm_85: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_987, permute_558);  view_987 = permute_558 = None
    view_988: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_84, [1, 16, 128, 256]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_327: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_44, view_988);  tangents_44 = view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_989: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_454: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_989, alias_73);  view_989 = None
    sum_89: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [-1], True)
    mul_455: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_73, sum_89);  alias_73 = sum_89 = None
    sub_99: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_73: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_99, primals_348);  sub_99 = primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_39: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1008, div_73, full_default_29);  slice_1008 = div_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_990: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_39, [16, 128, 128]);  where_39 = None
    bmm_86: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_559, view_990);  permute_559 = None
    bmm_87: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_990, permute_560);  view_990 = permute_560 = None
    view_991: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_86, [1, 16, 256, 128]);  bmm_86 = None
    view_992: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_87, [1, 16, 128, 256]);  bmm_87 = None
    permute_561: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_991, [0, 1, 3, 2]);  view_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_328: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_43, permute_561);  tangents_43 = permute_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_562: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_992, [0, 2, 1, 3]);  view_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_563: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_328, [0, 2, 1, 3]);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1376: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_562, 3, 0, 64)
    slice_1377: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_562, 3, 64, 256);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1378: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_563, 3, 0, 64)
    slice_1379: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_563, 3, 64, 256);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_456: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1376, view_571)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_993: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_456, [1, 128, 16, 32, 2]);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_28: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_993, 4, 0)
    select_29: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_993, 4, 1);  view_993 = None
    neg_71: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_28);  select_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_226: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_230: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_329: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_226, slice_scatter_230);  slice_scatter_226 = slice_scatter_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_457: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1376, view_572);  slice_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_330: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_329, mul_457);  add_329 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_458: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1378, view_571);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_994: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_458, [1, 128, 16, 32, 2]);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_30: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_994, 4, 0)
    select_31: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_994, 4, 1);  view_994 = None
    neg_72: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_30);  select_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_234: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_72, 3, 1, 9223372036854775807, 2);  neg_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_238: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_331: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_234, slice_scatter_238);  slice_scatter_234 = slice_scatter_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_459: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1378, view_572);  slice_1378 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_332: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_331, mul_459);  add_331 = mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_242: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1377, 3, 64, 9223372036854775807);  slice_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_246: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_330, 3, 0, 64);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_333: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_242, slice_scatter_246);  slice_scatter_242 = slice_scatter_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_250: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1379, 3, 64, 9223372036854775807);  slice_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_254: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_332, 3, 0, 64);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_334: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_250, slice_scatter_254);  slice_scatter_250 = slice_scatter_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_564: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_327, [0, 2, 1, 3]);  add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_232: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
    view_995: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_232, [1, 128, 4096]);  clone_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_996: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_334, [1, 128, 4096]);  add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_997: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_333, [1, 128, 4096]);  add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_998: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_995, [128, 4096]);  view_995 = None
    permute_565: "f32[4096, 128]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_204: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_565, view_562);  permute_565 = None
    permute_566: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_204, [1, 0]);  mm_204 = None
    mm_205: "f32[128, 4096]" = torch.ops.aten.mm.default(view_998, permute_567);  view_998 = permute_567 = None
    view_999: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_205, [1, 128, 4096]);  mm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_335: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_983, view_999);  view_983 = view_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_568: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1000: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_996, [128, 4096]);  view_996 = None
    permute_569: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1000, [1, 0])
    mm_206: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_569, view_562);  permute_569 = None
    permute_570: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_206, [1, 0]);  mm_206 = None
    mm_207: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1000, permute_571);  view_1000 = permute_571 = None
    view_1001: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_207, [1, 128, 4096]);  mm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_336: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_335, view_1001);  add_335 = view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_572: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1002: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_997, [128, 4096]);  view_997 = None
    permute_573: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1002, [1, 0])
    mm_208: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_573, view_562);  permute_573 = view_562 = None
    permute_574: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    mm_209: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1002, permute_575);  view_1002 = permute_575 = None
    view_1003: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_209, [1, 128, 4096]);  mm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_337: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_336, view_1003);  add_336 = view_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_576: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_461: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_337, primals_202);  primals_202 = None
    mul_462: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_461, 4096)
    sum_90: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
    mul_463: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_461, mul_200);  mul_461 = None
    sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
    mul_464: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, sum_91);  sum_91 = None
    sub_101: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_462, sum_90);  mul_462 = sum_90 = None
    sub_102: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_101, mul_464);  sub_101 = mul_464 = None
    mul_465: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_74, sub_102);  div_74 = sub_102 = None
    mul_466: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_337, mul_200);  mul_200 = None
    sum_92: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
    sum_93: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 1]);  add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_338: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_324, mul_465);  add_324 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1004: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_338, [128, 4096])
    mm_210: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1004, permute_577);  permute_577 = None
    permute_578: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1004, [1, 0])
    mm_211: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_578, view_560);  view_560 = None
    permute_579: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_94: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True)
    view_1005: "f32[4096]" = torch.ops.aten.reshape.default(sum_94, [4096]);  sum_94 = None
    permute_580: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_1006: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_210, [1, 128, 16384]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_467: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1006, mul_196);  mul_196 = None
    mul_468: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1006, add_157);  view_1006 = add_157 = None
    mul_469: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
    sub_103: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_469);  mul_469 = None
    mul_470: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_467, sub_103);  mul_467 = sub_103 = None
    mul_471: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_470, 0.7978845608028654);  mul_470 = None
    mul_472: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_471, 0.044715)
    pow_37: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_559, 2.0);  view_559 = None
    mul_473: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
    mul_474: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_339: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_471, mul_474);  mul_471 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_475: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_468, 0.5);  mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_340: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_339, mul_475);  add_339 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1007: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_340, [128, 16384]);  add_340 = None
    mm_212: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1007, permute_581);  permute_581 = None
    permute_582: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_213: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_582, view_534);  permute_582 = None
    permute_583: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_95: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
    view_1008: "f32[16384]" = torch.ops.aten.reshape.default(sum_95, [16384]);  sum_95 = None
    permute_584: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_1009: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_212, [1, 128, 4096]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_214: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_578, view_556);  permute_578 = view_556 = None
    permute_586: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_214, [1, 0]);  mm_214 = None
    mm_215: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1004, permute_587);  view_1004 = permute_587 = None
    view_1011: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_215, [1, 128, 4096]);  mm_215 = None
    permute_588: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1012: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1011, [1, 128, 16, 256]);  view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_589: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1012, [0, 2, 1, 3]);  view_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1013: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_589, [16, 128, 256]);  permute_589 = None
    bmm_88: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_590, view_1013);  permute_590 = None
    bmm_89: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1013, permute_591);  view_1013 = permute_591 = None
    view_1014: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_88, [1, 16, 128, 256]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_341: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_42, view_1014);  tangents_42 = view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1015: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_476: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1015, alias_75);  view_1015 = None
    sum_96: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
    mul_477: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_75, sum_96);  alias_75 = sum_96 = None
    sub_104: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_75: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_104, primals_345);  sub_104 = primals_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_40: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_960, div_75, full_default_29);  slice_960 = div_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1016: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_40, [16, 128, 128]);  where_40 = None
    bmm_90: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_592, view_1016);  permute_592 = None
    bmm_91: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1016, permute_593);  view_1016 = permute_593 = None
    view_1017: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_90, [1, 16, 256, 128]);  bmm_90 = None
    view_1018: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_91, [1, 16, 128, 256]);  bmm_91 = None
    permute_594: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1017, [0, 1, 3, 2]);  view_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_342: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_41, permute_594);  tangents_41 = permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_595: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1018, [0, 2, 1, 3]);  view_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_596: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_342, [0, 2, 1, 3]);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1380: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_595, 3, 0, 64)
    slice_1381: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_595, 3, 64, 256);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1382: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_596, 3, 0, 64)
    slice_1383: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_596, 3, 64, 256);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_478: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1380, view_543)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1019: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_478, [1, 128, 16, 32, 2]);  mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_32: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1019, 4, 0)
    select_33: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1019, 4, 1);  view_1019 = None
    neg_73: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_32);  select_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_258: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_262: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_343: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_258, slice_scatter_262);  slice_scatter_258 = slice_scatter_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_479: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1380, view_544);  slice_1380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_344: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_343, mul_479);  add_343 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_480: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1382, view_543);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1020: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_480, [1, 128, 16, 32, 2]);  mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_34: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1020, 4, 0)
    select_35: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1020, 4, 1);  view_1020 = None
    neg_74: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_34);  select_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_266: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_270: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_345: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_266, slice_scatter_270);  slice_scatter_266 = slice_scatter_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_481: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1382, view_544);  slice_1382 = view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_346: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_345, mul_481);  add_345 = mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_274: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1381, 3, 64, 9223372036854775807);  slice_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_278: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_344, 3, 0, 64);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_347: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_274, slice_scatter_278);  slice_scatter_274 = slice_scatter_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_282: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1383, 3, 64, 9223372036854775807);  slice_1383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_286: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_346, 3, 0, 64);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_348: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_282, slice_scatter_286);  slice_scatter_282 = slice_scatter_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_597: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_341, [0, 2, 1, 3]);  add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_233: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
    view_1021: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_233, [1, 128, 4096]);  clone_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1022: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_348, [1, 128, 4096]);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1023: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_347, [1, 128, 4096]);  add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1024: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1021, [128, 4096]);  view_1021 = None
    permute_598: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1024, [1, 0])
    mm_216: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_598, view_534);  permute_598 = None
    permute_599: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_216, [1, 0]);  mm_216 = None
    mm_217: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1024, permute_600);  view_1024 = permute_600 = None
    view_1025: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_217, [1, 128, 4096]);  mm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_349: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1009, view_1025);  view_1009 = view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_601: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1026: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1022, [128, 4096]);  view_1022 = None
    permute_602: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1026, [1, 0])
    mm_218: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_602, view_534);  permute_602 = None
    permute_603: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_218, [1, 0]);  mm_218 = None
    mm_219: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1026, permute_604);  view_1026 = permute_604 = None
    view_1027: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_219, [1, 128, 4096]);  mm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_350: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_349, view_1027);  add_349 = view_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_605: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1028: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1023, [128, 4096]);  view_1023 = None
    permute_606: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1028, [1, 0])
    mm_220: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_606, view_534);  permute_606 = view_534 = None
    permute_607: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_220, [1, 0]);  mm_220 = None
    mm_221: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1028, permute_608);  view_1028 = permute_608 = None
    view_1029: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_221, [1, 128, 4096]);  mm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_351: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_350, view_1029);  add_350 = view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_609: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_483: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_351, primals_192);  primals_192 = None
    mul_484: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_483, 4096)
    sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_483, mul_190);  mul_483 = None
    sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, sum_98);  sum_98 = None
    sub_106: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_484, sum_97);  mul_484 = sum_97 = None
    sub_107: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_106, mul_486);  sub_106 = mul_486 = None
    mul_487: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_76, sub_107);  div_76 = sub_107 = None
    mul_488: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_351, mul_190);  mul_190 = None
    sum_99: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 1]);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_352: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_338, mul_487);  add_338 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1030: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_352, [128, 4096])
    mm_222: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1030, permute_610);  permute_610 = None
    permute_611: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1030, [1, 0])
    mm_223: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_611, view_532);  view_532 = None
    permute_612: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_101: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1030, [0], True)
    view_1031: "f32[4096]" = torch.ops.aten.reshape.default(sum_101, [4096]);  sum_101 = None
    permute_613: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_1032: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_222, [1, 128, 16384]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_489: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1032, mul_186);  mul_186 = None
    mul_490: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1032, add_149);  view_1032 = add_149 = None
    mul_491: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
    sub_108: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_491);  mul_491 = None
    mul_492: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_489, sub_108);  mul_489 = sub_108 = None
    mul_493: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_492, 0.7978845608028654);  mul_492 = None
    mul_494: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_493, 0.044715)
    pow_38: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_531, 2.0);  view_531 = None
    mul_495: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
    mul_496: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_353: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_493, mul_496);  mul_493 = mul_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_497: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_490, 0.5);  mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_354: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_353, mul_497);  add_353 = mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1033: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_354, [128, 16384]);  add_354 = None
    mm_224: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1033, permute_614);  permute_614 = None
    permute_615: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1033, [1, 0])
    mm_225: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_615, view_506);  permute_615 = None
    permute_616: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_102: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1033, [0], True);  view_1033 = None
    view_1034: "f32[16384]" = torch.ops.aten.reshape.default(sum_102, [16384]);  sum_102 = None
    permute_617: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_616, [1, 0]);  permute_616 = None
    view_1035: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_224, [1, 128, 4096]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_226: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_611, view_528);  permute_611 = view_528 = None
    permute_619: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_226, [1, 0]);  mm_226 = None
    mm_227: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1030, permute_620);  view_1030 = permute_620 = None
    view_1037: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_227, [1, 128, 4096]);  mm_227 = None
    permute_621: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1038: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1037, [1, 128, 16, 256]);  view_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_622: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1038, [0, 2, 1, 3]);  view_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1039: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_622, [16, 128, 256]);  permute_622 = None
    bmm_92: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_623, view_1039);  permute_623 = None
    bmm_93: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1039, permute_624);  view_1039 = permute_624 = None
    view_1040: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_92, [1, 16, 128, 256]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_355: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_40, view_1040);  tangents_40 = view_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1041: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_498: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1041, alias_77);  view_1041 = None
    sum_103: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [-1], True)
    mul_499: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_77, sum_103);  alias_77 = sum_103 = None
    sub_109: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_77: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_109, primals_342);  sub_109 = primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_41: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_912, div_77, full_default_29);  slice_912 = div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1042: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_41, [16, 128, 128]);  where_41 = None
    bmm_94: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_625, view_1042);  permute_625 = None
    bmm_95: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1042, permute_626);  view_1042 = permute_626 = None
    view_1043: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_94, [1, 16, 256, 128]);  bmm_94 = None
    view_1044: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_95, [1, 16, 128, 256]);  bmm_95 = None
    permute_627: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1043, [0, 1, 3, 2]);  view_1043 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_356: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_39, permute_627);  tangents_39 = permute_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_628: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1044, [0, 2, 1, 3]);  view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_629: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_356, [0, 2, 1, 3]);  add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1384: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_628, 3, 0, 64)
    slice_1385: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_628, 3, 64, 256);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1386: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_629, 3, 0, 64)
    slice_1387: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_629, 3, 64, 256);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_500: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1384, view_515)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1045: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_500, [1, 128, 16, 32, 2]);  mul_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_36: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1045, 4, 0)
    select_37: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1045, 4, 1);  view_1045 = None
    neg_75: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_36);  select_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_290: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_75, 3, 1, 9223372036854775807, 2);  neg_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_294: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_357: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_290, slice_scatter_294);  slice_scatter_290 = slice_scatter_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_501: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1384, view_516);  slice_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_358: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_357, mul_501);  add_357 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_502: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1386, view_515);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1046: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_502, [1, 128, 16, 32, 2]);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_38: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1046, 4, 0)
    select_39: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1046, 4, 1);  view_1046 = None
    neg_76: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_38);  select_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_298: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_302: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_359: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_298, slice_scatter_302);  slice_scatter_298 = slice_scatter_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_503: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1386, view_516);  slice_1386 = view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_360: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_359, mul_503);  add_359 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_306: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1385, 3, 64, 9223372036854775807);  slice_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_310: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_358, 3, 0, 64);  add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_361: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_306, slice_scatter_310);  slice_scatter_306 = slice_scatter_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_314: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1387, 3, 64, 9223372036854775807);  slice_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_318: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_360, 3, 0, 64);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_362: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_314, slice_scatter_318);  slice_scatter_314 = slice_scatter_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_630: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_355, [0, 2, 1, 3]);  add_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_234: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
    view_1047: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_234, [1, 128, 4096]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1048: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_362, [1, 128, 4096]);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1049: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_361, [1, 128, 4096]);  add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1050: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1047, [128, 4096]);  view_1047 = None
    permute_631: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1050, [1, 0])
    mm_228: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_631, view_506);  permute_631 = None
    permute_632: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_228, [1, 0]);  mm_228 = None
    mm_229: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1050, permute_633);  view_1050 = permute_633 = None
    view_1051: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_229, [1, 128, 4096]);  mm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_363: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1035, view_1051);  view_1035 = view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_634: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1052: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1048, [128, 4096]);  view_1048 = None
    permute_635: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1052, [1, 0])
    mm_230: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_635, view_506);  permute_635 = None
    permute_636: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_230, [1, 0]);  mm_230 = None
    mm_231: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1052, permute_637);  view_1052 = permute_637 = None
    view_1053: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_231, [1, 128, 4096]);  mm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_364: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_363, view_1053);  add_363 = view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_638: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1054: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1049, [128, 4096]);  view_1049 = None
    permute_639: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1054, [1, 0])
    mm_232: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_639, view_506);  permute_639 = view_506 = None
    permute_640: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    mm_233: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1054, permute_641);  view_1054 = permute_641 = None
    view_1055: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_233, [1, 128, 4096]);  mm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_365: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_364, view_1055);  add_364 = view_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_642: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_505: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_365, primals_182);  primals_182 = None
    mul_506: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_505, 4096)
    sum_104: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True)
    mul_507: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_505, mul_180);  mul_505 = None
    sum_105: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True);  mul_507 = None
    mul_508: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, sum_105);  sum_105 = None
    sub_111: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_506, sum_104);  mul_506 = sum_104 = None
    sub_112: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_111, mul_508);  sub_111 = mul_508 = None
    mul_509: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_78, sub_112);  div_78 = sub_112 = None
    mul_510: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_365, mul_180);  mul_180 = None
    sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1]);  mul_510 = None
    sum_107: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 1]);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_366: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_352, mul_509);  add_352 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1056: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_366, [128, 4096])
    mm_234: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1056, permute_643);  permute_643 = None
    permute_644: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1056, [1, 0])
    mm_235: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_644, view_504);  view_504 = None
    permute_645: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1056, [0], True)
    view_1057: "f32[4096]" = torch.ops.aten.reshape.default(sum_108, [4096]);  sum_108 = None
    permute_646: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_645, [1, 0]);  permute_645 = None
    view_1058: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_234, [1, 128, 16384]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_511: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1058, mul_176);  mul_176 = None
    mul_512: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1058, add_141);  view_1058 = add_141 = None
    mul_513: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
    sub_113: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_513);  mul_513 = None
    mul_514: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_511, sub_113);  mul_511 = sub_113 = None
    mul_515: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_514, 0.7978845608028654);  mul_514 = None
    mul_516: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_515, 0.044715)
    pow_39: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 2.0);  view_503 = None
    mul_517: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
    mul_518: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_367: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_515, mul_518);  mul_515 = mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_519: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_512, 0.5);  mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_368: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_367, mul_519);  add_367 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1059: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_368, [128, 16384]);  add_368 = None
    mm_236: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1059, permute_647);  permute_647 = None
    permute_648: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1059, [1, 0])
    mm_237: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_648, view_478);  permute_648 = None
    permute_649: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_109: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
    view_1060: "f32[16384]" = torch.ops.aten.reshape.default(sum_109, [16384]);  sum_109 = None
    permute_650: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
    view_1061: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_236, [1, 128, 4096]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_238: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_644, view_500);  permute_644 = view_500 = None
    permute_652: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_238, [1, 0]);  mm_238 = None
    mm_239: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1056, permute_653);  view_1056 = permute_653 = None
    view_1063: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_239, [1, 128, 4096]);  mm_239 = None
    permute_654: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1064: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1063, [1, 128, 16, 256]);  view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_655: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1064, [0, 2, 1, 3]);  view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1065: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_655, [16, 128, 256]);  permute_655 = None
    bmm_96: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_656, view_1065);  permute_656 = None
    bmm_97: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1065, permute_657);  view_1065 = permute_657 = None
    view_1066: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_96, [1, 16, 128, 256]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_369: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_38, view_1066);  tangents_38 = view_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1067: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_520: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1067, alias_79);  view_1067 = None
    sum_110: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [-1], True)
    mul_521: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_79, sum_110);  alias_79 = sum_110 = None
    sub_114: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_79: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_114, primals_339);  sub_114 = primals_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_42: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_864, div_79, full_default_29);  slice_864 = div_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1068: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_42, [16, 128, 128]);  where_42 = None
    bmm_98: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_658, view_1068);  permute_658 = None
    bmm_99: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1068, permute_659);  view_1068 = permute_659 = None
    view_1069: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_98, [1, 16, 256, 128]);  bmm_98 = None
    view_1070: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_99, [1, 16, 128, 256]);  bmm_99 = None
    permute_660: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1069, [0, 1, 3, 2]);  view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_370: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_37, permute_660);  tangents_37 = permute_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_661: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1070, [0, 2, 1, 3]);  view_1070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_662: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_370, [0, 2, 1, 3]);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1388: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_661, 3, 0, 64)
    slice_1389: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_661, 3, 64, 256);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1390: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_662, 3, 0, 64)
    slice_1391: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_662, 3, 64, 256);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_522: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1388, view_487)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1071: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_522, [1, 128, 16, 32, 2]);  mul_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_40: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1071, 4, 0)
    select_41: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1071, 4, 1);  view_1071 = None
    neg_77: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_40);  select_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_322: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_326: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_371: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_322, slice_scatter_326);  slice_scatter_322 = slice_scatter_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_523: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1388, view_488);  slice_1388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_372: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_371, mul_523);  add_371 = mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_524: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1390, view_487);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1072: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_524, [1, 128, 16, 32, 2]);  mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_42: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1072, 4, 0)
    select_43: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1072, 4, 1);  view_1072 = None
    neg_78: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_42);  select_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_330: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_78, 3, 1, 9223372036854775807, 2);  neg_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_334: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_373: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_330, slice_scatter_334);  slice_scatter_330 = slice_scatter_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_525: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1390, view_488);  slice_1390 = view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_374: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_373, mul_525);  add_373 = mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_338: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1389, 3, 64, 9223372036854775807);  slice_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_342: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_372, 3, 0, 64);  add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_375: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_338, slice_scatter_342);  slice_scatter_338 = slice_scatter_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_346: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1391, 3, 64, 9223372036854775807);  slice_1391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_350: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_374, 3, 0, 64);  add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_376: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_346, slice_scatter_350);  slice_scatter_346 = slice_scatter_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_663: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_369, [0, 2, 1, 3]);  add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_235: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
    view_1073: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_235, [1, 128, 4096]);  clone_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1074: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_376, [1, 128, 4096]);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1075: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_375, [1, 128, 4096]);  add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1076: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1073, [128, 4096]);  view_1073 = None
    permute_664: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1076, [1, 0])
    mm_240: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_664, view_478);  permute_664 = None
    permute_665: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_240, [1, 0]);  mm_240 = None
    mm_241: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1076, permute_666);  view_1076 = permute_666 = None
    view_1077: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_241, [1, 128, 4096]);  mm_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_377: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1061, view_1077);  view_1061 = view_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_667: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1078: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1074, [128, 4096]);  view_1074 = None
    permute_668: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1078, [1, 0])
    mm_242: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_668, view_478);  permute_668 = None
    permute_669: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_242, [1, 0]);  mm_242 = None
    mm_243: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1078, permute_670);  view_1078 = permute_670 = None
    view_1079: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_378: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_377, view_1079);  add_377 = view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_671: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1080: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1075, [128, 4096]);  view_1075 = None
    permute_672: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1080, [1, 0])
    mm_244: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_672, view_478);  permute_672 = view_478 = None
    permute_673: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    mm_245: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1080, permute_674);  view_1080 = permute_674 = None
    view_1081: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_245, [1, 128, 4096]);  mm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_379: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_378, view_1081);  add_378 = view_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_675: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_527: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_379, primals_172);  primals_172 = None
    mul_528: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_527, 4096)
    sum_111: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
    mul_529: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_527, mul_170);  mul_527 = None
    sum_112: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
    mul_530: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, sum_112);  sum_112 = None
    sub_116: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_528, sum_111);  mul_528 = sum_111 = None
    sub_117: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_116, mul_530);  sub_116 = mul_530 = None
    mul_531: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_80, sub_117);  div_80 = sub_117 = None
    mul_532: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_379, mul_170);  mul_170 = None
    sum_113: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
    sum_114: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_379, [0, 1]);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_380: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_366, mul_531);  add_366 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1082: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_380, [128, 4096])
    mm_246: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1082, permute_676);  permute_676 = None
    permute_677: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1082, [1, 0])
    mm_247: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_677, view_476);  view_476 = None
    permute_678: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1082, [0], True)
    view_1083: "f32[4096]" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
    permute_679: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_678, [1, 0]);  permute_678 = None
    view_1084: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_246, [1, 128, 16384]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_533: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1084, mul_166);  mul_166 = None
    mul_534: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1084, add_133);  view_1084 = add_133 = None
    mul_535: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
    sub_118: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_535);  mul_535 = None
    mul_536: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_533, sub_118);  mul_533 = sub_118 = None
    mul_537: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_536, 0.7978845608028654);  mul_536 = None
    mul_538: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_537, 0.044715)
    pow_40: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_475, 2.0);  view_475 = None
    mul_539: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
    mul_540: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_381: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_537, mul_540);  mul_537 = mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_541: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_534, 0.5);  mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_382: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_381, mul_541);  add_381 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1085: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_382, [128, 16384]);  add_382 = None
    mm_248: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1085, permute_680);  permute_680 = None
    permute_681: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1085, [1, 0])
    mm_249: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_681, view_450);  permute_681 = None
    permute_682: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_116: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1085, [0], True);  view_1085 = None
    view_1086: "f32[16384]" = torch.ops.aten.reshape.default(sum_116, [16384]);  sum_116 = None
    permute_683: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    view_1087: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_248, [1, 128, 4096]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_250: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_677, view_472);  permute_677 = view_472 = None
    permute_685: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_250, [1, 0]);  mm_250 = None
    mm_251: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1082, permute_686);  view_1082 = permute_686 = None
    view_1089: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_251, [1, 128, 4096]);  mm_251 = None
    permute_687: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_685, [1, 0]);  permute_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1090: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1089, [1, 128, 16, 256]);  view_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_688: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1090, [0, 2, 1, 3]);  view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1091: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_688, [16, 128, 256]);  permute_688 = None
    bmm_100: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_689, view_1091);  permute_689 = None
    bmm_101: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1091, permute_690);  view_1091 = permute_690 = None
    view_1092: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_100, [1, 16, 128, 256]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_383: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_36, view_1092);  tangents_36 = view_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1093: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_542: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1093, alias_81);  view_1093 = None
    sum_117: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [-1], True)
    mul_543: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_81, sum_117);  alias_81 = sum_117 = None
    sub_119: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_81: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_119, primals_336);  sub_119 = primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_43: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_816, div_81, full_default_29);  slice_816 = div_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1094: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_43, [16, 128, 128]);  where_43 = None
    bmm_102: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_691, view_1094);  permute_691 = None
    bmm_103: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1094, permute_692);  view_1094 = permute_692 = None
    view_1095: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_102, [1, 16, 256, 128]);  bmm_102 = None
    view_1096: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_103, [1, 16, 128, 256]);  bmm_103 = None
    permute_693: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1095, [0, 1, 3, 2]);  view_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_384: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_35, permute_693);  tangents_35 = permute_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_694: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_695: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_384, [0, 2, 1, 3]);  add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1392: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_694, 3, 0, 64)
    slice_1393: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_694, 3, 64, 256);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1394: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_695, 3, 0, 64)
    slice_1395: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_695, 3, 64, 256);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_544: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1392, view_459)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1097: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_544, [1, 128, 16, 32, 2]);  mul_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_44: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1097, 4, 0)
    select_45: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1097, 4, 1);  view_1097 = None
    neg_79: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_44);  select_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_354: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_358: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_385: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_354, slice_scatter_358);  slice_scatter_354 = slice_scatter_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_545: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1392, view_460);  slice_1392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_386: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_385, mul_545);  add_385 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_546: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1394, view_459);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1098: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_546, [1, 128, 16, 32, 2]);  mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_46: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1098, 4, 0)
    select_47: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1098, 4, 1);  view_1098 = None
    neg_80: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_46);  select_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_362: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_366: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_387: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_362, slice_scatter_366);  slice_scatter_362 = slice_scatter_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_547: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1394, view_460);  slice_1394 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_388: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_387, mul_547);  add_387 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_370: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1393, 3, 64, 9223372036854775807);  slice_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_374: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_386, 3, 0, 64);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_389: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_370, slice_scatter_374);  slice_scatter_370 = slice_scatter_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_378: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1395, 3, 64, 9223372036854775807);  slice_1395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_382: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_388, 3, 0, 64);  add_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_390: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_378, slice_scatter_382);  slice_scatter_378 = slice_scatter_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_696: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_383, [0, 2, 1, 3]);  add_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_236: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
    view_1099: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_236, [1, 128, 4096]);  clone_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1100: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_390, [1, 128, 4096]);  add_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1101: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_389, [1, 128, 4096]);  add_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1102: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1099, [128, 4096]);  view_1099 = None
    permute_697: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1102, [1, 0])
    mm_252: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_697, view_450);  permute_697 = None
    permute_698: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_252, [1, 0]);  mm_252 = None
    mm_253: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1102, permute_699);  view_1102 = permute_699 = None
    view_1103: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_253, [1, 128, 4096]);  mm_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_391: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1087, view_1103);  view_1087 = view_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_700: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1104: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1100, [128, 4096]);  view_1100 = None
    permute_701: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1104, [1, 0])
    mm_254: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_701, view_450);  permute_701 = None
    permute_702: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_254, [1, 0]);  mm_254 = None
    mm_255: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1104, permute_703);  view_1104 = permute_703 = None
    view_1105: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_392: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_391, view_1105);  add_391 = view_1105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_704: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1106: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1101, [128, 4096]);  view_1101 = None
    permute_705: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1106, [1, 0])
    mm_256: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_705, view_450);  permute_705 = view_450 = None
    permute_706: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    mm_257: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1106, permute_707);  view_1106 = permute_707 = None
    view_1107: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_257, [1, 128, 4096]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_393: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_392, view_1107);  add_392 = view_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_708: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_549: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_393, primals_162);  primals_162 = None
    mul_550: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_549, 4096)
    sum_118: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True)
    mul_551: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_549, mul_160);  mul_549 = None
    sum_119: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True);  mul_551 = None
    mul_552: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, sum_119);  sum_119 = None
    sub_121: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_550, sum_118);  mul_550 = sum_118 = None
    sub_122: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_121, mul_552);  sub_121 = mul_552 = None
    mul_553: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_82, sub_122);  div_82 = sub_122 = None
    mul_554: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_393, mul_160);  mul_160 = None
    sum_120: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1]);  mul_554 = None
    sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_393, [0, 1]);  add_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_394: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_380, mul_553);  add_380 = mul_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1108: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_394, [128, 4096])
    mm_258: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1108, permute_709);  permute_709 = None
    permute_710: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1108, [1, 0])
    mm_259: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_710, view_448);  view_448 = None
    permute_711: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_122: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1108, [0], True)
    view_1109: "f32[4096]" = torch.ops.aten.reshape.default(sum_122, [4096]);  sum_122 = None
    permute_712: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_711, [1, 0]);  permute_711 = None
    view_1110: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_258, [1, 128, 16384]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_555: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1110, mul_156);  mul_156 = None
    mul_556: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1110, add_125);  view_1110 = add_125 = None
    mul_557: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
    sub_123: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_557);  mul_557 = None
    mul_558: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_555, sub_123);  mul_555 = sub_123 = None
    mul_559: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_558, 0.7978845608028654);  mul_558 = None
    mul_560: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_559, 0.044715)
    pow_41: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_447, 2.0);  view_447 = None
    mul_561: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
    mul_562: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_560, mul_561);  mul_560 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_395: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_559, mul_562);  mul_559 = mul_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_563: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_556, 0.5);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_396: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_395, mul_563);  add_395 = mul_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1111: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_396, [128, 16384]);  add_396 = None
    mm_260: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1111, permute_713);  permute_713 = None
    permute_714: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1111, [1, 0])
    mm_261: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_714, view_422);  permute_714 = None
    permute_715: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_123: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1111, [0], True);  view_1111 = None
    view_1112: "f32[16384]" = torch.ops.aten.reshape.default(sum_123, [16384]);  sum_123 = None
    permute_716: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    view_1113: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_260, [1, 128, 4096]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_262: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_710, view_444);  permute_710 = view_444 = None
    permute_718: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_262, [1, 0]);  mm_262 = None
    mm_263: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1108, permute_719);  view_1108 = permute_719 = None
    view_1115: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_263, [1, 128, 4096]);  mm_263 = None
    permute_720: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1116: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1115, [1, 128, 16, 256]);  view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_721: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1116, [0, 2, 1, 3]);  view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1117: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_721, [16, 128, 256]);  permute_721 = None
    bmm_104: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_722, view_1117);  permute_722 = None
    bmm_105: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1117, permute_723);  view_1117 = permute_723 = None
    view_1118: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_104, [1, 16, 128, 256]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_397: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_34, view_1118);  tangents_34 = view_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1119: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_564: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1119, alias_83);  view_1119 = None
    sum_124: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [-1], True)
    mul_565: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_83, sum_124);  alias_83 = sum_124 = None
    sub_124: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_83: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_124, primals_333);  sub_124 = primals_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_44: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_768, div_83, full_default_29);  slice_768 = div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1120: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_44, [16, 128, 128]);  where_44 = None
    bmm_106: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_724, view_1120);  permute_724 = None
    bmm_107: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1120, permute_725);  view_1120 = permute_725 = None
    view_1121: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_106, [1, 16, 256, 128]);  bmm_106 = None
    view_1122: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_107, [1, 16, 128, 256]);  bmm_107 = None
    permute_726: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1121, [0, 1, 3, 2]);  view_1121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_398: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_33, permute_726);  tangents_33 = permute_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_727: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1122, [0, 2, 1, 3]);  view_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_728: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_398, [0, 2, 1, 3]);  add_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1396: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_727, 3, 0, 64)
    slice_1397: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_727, 3, 64, 256);  permute_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1398: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_728, 3, 0, 64)
    slice_1399: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_728, 3, 64, 256);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_566: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1396, view_431)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1123: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_566, [1, 128, 16, 32, 2]);  mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_48: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1123, 4, 0)
    select_49: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1123, 4, 1);  view_1123 = None
    neg_81: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_48);  select_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_386: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_81, 3, 1, 9223372036854775807, 2);  neg_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_390: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_399: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_386, slice_scatter_390);  slice_scatter_386 = slice_scatter_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_567: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1396, view_432);  slice_1396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_400: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_399, mul_567);  add_399 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_568: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1398, view_431);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1124: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_568, [1, 128, 16, 32, 2]);  mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_50: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1124, 4, 0)
    select_51: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1124, 4, 1);  view_1124 = None
    neg_82: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_50);  select_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_394: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_398: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_401: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_394, slice_scatter_398);  slice_scatter_394 = slice_scatter_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_569: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1398, view_432);  slice_1398 = view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_402: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_401, mul_569);  add_401 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_402: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1397, 3, 64, 9223372036854775807);  slice_1397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_406: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_400, 3, 0, 64);  add_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_403: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_402, slice_scatter_406);  slice_scatter_402 = slice_scatter_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_410: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1399, 3, 64, 9223372036854775807);  slice_1399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_414: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_402, 3, 0, 64);  add_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_404: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_410, slice_scatter_414);  slice_scatter_410 = slice_scatter_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_729: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_397, [0, 2, 1, 3]);  add_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_237: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
    view_1125: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_237, [1, 128, 4096]);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1126: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_404, [1, 128, 4096]);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1127: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_403, [1, 128, 4096]);  add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1128: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1125, [128, 4096]);  view_1125 = None
    permute_730: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1128, [1, 0])
    mm_264: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_730, view_422);  permute_730 = None
    permute_731: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_264, [1, 0]);  mm_264 = None
    mm_265: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1128, permute_732);  view_1128 = permute_732 = None
    view_1129: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_265, [1, 128, 4096]);  mm_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_405: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1113, view_1129);  view_1113 = view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_733: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1130: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1126, [128, 4096]);  view_1126 = None
    permute_734: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1130, [1, 0])
    mm_266: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_734, view_422);  permute_734 = None
    permute_735: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_266, [1, 0]);  mm_266 = None
    mm_267: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1130, permute_736);  view_1130 = permute_736 = None
    view_1131: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_406: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_405, view_1131);  add_405 = view_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_737: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1132: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1127, [128, 4096]);  view_1127 = None
    permute_738: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
    mm_268: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_738, view_422);  permute_738 = view_422 = None
    permute_739: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    mm_269: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1132, permute_740);  view_1132 = permute_740 = None
    view_1133: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_269, [1, 128, 4096]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_407: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_406, view_1133);  add_406 = view_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_741: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_739, [1, 0]);  permute_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_571: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_407, primals_152);  primals_152 = None
    mul_572: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_571, 4096)
    sum_125: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True)
    mul_573: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_571, mul_150);  mul_571 = None
    sum_126: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
    mul_574: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, sum_126);  sum_126 = None
    sub_126: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_572, sum_125);  mul_572 = sum_125 = None
    sub_127: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_126, mul_574);  sub_126 = mul_574 = None
    mul_575: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_84, sub_127);  div_84 = sub_127 = None
    mul_576: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_407, mul_150);  mul_150 = None
    sum_127: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1]);  mul_576 = None
    sum_128: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_407, [0, 1]);  add_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_408: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_394, mul_575);  add_394 = mul_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1134: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_408, [128, 4096])
    mm_270: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1134, permute_742);  permute_742 = None
    permute_743: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_271: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_743, view_420);  view_420 = None
    permute_744: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_129: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1134, [0], True)
    view_1135: "f32[4096]" = torch.ops.aten.reshape.default(sum_129, [4096]);  sum_129 = None
    permute_745: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_744, [1, 0]);  permute_744 = None
    view_1136: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_270, [1, 128, 16384]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_577: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1136, mul_146);  mul_146 = None
    mul_578: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1136, add_117);  view_1136 = add_117 = None
    mul_579: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
    sub_128: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_579);  mul_579 = None
    mul_580: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_577, sub_128);  mul_577 = sub_128 = None
    mul_581: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_580, 0.7978845608028654);  mul_580 = None
    mul_582: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_581, 0.044715)
    pow_42: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_419, 2.0);  view_419 = None
    mul_583: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
    mul_584: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_409: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_581, mul_584);  mul_581 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_585: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_578, 0.5);  mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_410: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_409, mul_585);  add_409 = mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1137: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_410, [128, 16384]);  add_410 = None
    mm_272: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1137, permute_746);  permute_746 = None
    permute_747: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1137, [1, 0])
    mm_273: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_747, view_394);  permute_747 = None
    permute_748: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_130: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1137, [0], True);  view_1137 = None
    view_1138: "f32[16384]" = torch.ops.aten.reshape.default(sum_130, [16384]);  sum_130 = None
    permute_749: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_1139: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_272, [1, 128, 4096]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_274: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_743, view_416);  permute_743 = view_416 = None
    permute_751: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_274, [1, 0]);  mm_274 = None
    mm_275: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1134, permute_752);  view_1134 = permute_752 = None
    view_1141: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_275, [1, 128, 4096]);  mm_275 = None
    permute_753: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_751, [1, 0]);  permute_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1142: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1141, [1, 128, 16, 256]);  view_1141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_754: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1143: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_754, [16, 128, 256]);  permute_754 = None
    bmm_108: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_755, view_1143);  permute_755 = None
    bmm_109: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1143, permute_756);  view_1143 = permute_756 = None
    view_1144: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_108, [1, 16, 128, 256]);  bmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_411: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_32, view_1144);  tangents_32 = view_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1145: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_586: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1145, alias_85);  view_1145 = None
    sum_131: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [-1], True)
    mul_587: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_85, sum_131);  alias_85 = sum_131 = None
    sub_129: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_85: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_129, primals_330);  sub_129 = primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_45: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_720, div_85, full_default_29);  slice_720 = div_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1146: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_45, [16, 128, 128]);  where_45 = None
    bmm_110: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_757, view_1146);  permute_757 = None
    bmm_111: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1146, permute_758);  view_1146 = permute_758 = None
    view_1147: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_110, [1, 16, 256, 128]);  bmm_110 = None
    view_1148: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_111, [1, 16, 128, 256]);  bmm_111 = None
    permute_759: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1147, [0, 1, 3, 2]);  view_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_412: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_31, permute_759);  tangents_31 = permute_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_760: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1148, [0, 2, 1, 3]);  view_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_761: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_412, [0, 2, 1, 3]);  add_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1400: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_760, 3, 0, 64)
    slice_1401: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_760, 3, 64, 256);  permute_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1402: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_761, 3, 0, 64)
    slice_1403: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_761, 3, 64, 256);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_588: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1400, view_403)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1149: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_588, [1, 128, 16, 32, 2]);  mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_52: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1149, 4, 0)
    select_53: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1149, 4, 1);  view_1149 = None
    neg_83: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_52);  select_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_418: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_422: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_413: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_418, slice_scatter_422);  slice_scatter_418 = slice_scatter_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_589: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1400, view_404);  slice_1400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_414: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_413, mul_589);  add_413 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_590: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1402, view_403);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1150: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_590, [1, 128, 16, 32, 2]);  mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_54: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1150, 4, 0)
    select_55: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1150, 4, 1);  view_1150 = None
    neg_84: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_54);  select_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_426: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_84, 3, 1, 9223372036854775807, 2);  neg_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_430: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_415: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_426, slice_scatter_430);  slice_scatter_426 = slice_scatter_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_591: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1402, view_404);  slice_1402 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_416: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_415, mul_591);  add_415 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_434: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1401, 3, 64, 9223372036854775807);  slice_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_438: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_414, 3, 0, 64);  add_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_417: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_434, slice_scatter_438);  slice_scatter_434 = slice_scatter_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_442: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1403, 3, 64, 9223372036854775807);  slice_1403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_446: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_416, 3, 0, 64);  add_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_418: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_442, slice_scatter_446);  slice_scatter_442 = slice_scatter_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_762: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_411, [0, 2, 1, 3]);  add_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_238: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
    view_1151: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_238, [1, 128, 4096]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1152: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_418, [1, 128, 4096]);  add_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1153: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_417, [1, 128, 4096]);  add_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1154: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1151, [128, 4096]);  view_1151 = None
    permute_763: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1154, [1, 0])
    mm_276: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_763, view_394);  permute_763 = None
    permute_764: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_276, [1, 0]);  mm_276 = None
    mm_277: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1154, permute_765);  view_1154 = permute_765 = None
    view_1155: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_277, [1, 128, 4096]);  mm_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_419: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1139, view_1155);  view_1139 = view_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_766: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_764, [1, 0]);  permute_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1156: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1152, [128, 4096]);  view_1152 = None
    permute_767: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1156, [1, 0])
    mm_278: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_767, view_394);  permute_767 = None
    permute_768: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_278, [1, 0]);  mm_278 = None
    mm_279: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1156, permute_769);  view_1156 = permute_769 = None
    view_1157: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_420: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_419, view_1157);  add_419 = view_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_770: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_768, [1, 0]);  permute_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1158: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1153, [128, 4096]);  view_1153 = None
    permute_771: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1158, [1, 0])
    mm_280: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_771, view_394);  permute_771 = view_394 = None
    permute_772: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_280, [1, 0]);  mm_280 = None
    mm_281: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1158, permute_773);  view_1158 = permute_773 = None
    view_1159: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_281, [1, 128, 4096]);  mm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_421: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_420, view_1159);  add_420 = view_1159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_774: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_772, [1, 0]);  permute_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_593: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_421, primals_142);  primals_142 = None
    mul_594: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_593, 4096)
    sum_132: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True)
    mul_595: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_593, mul_140);  mul_593 = None
    sum_133: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True);  mul_595 = None
    mul_596: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, sum_133);  sum_133 = None
    sub_131: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_594, sum_132);  mul_594 = sum_132 = None
    sub_132: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_131, mul_596);  sub_131 = mul_596 = None
    mul_597: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_86, sub_132);  div_86 = sub_132 = None
    mul_598: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_421, mul_140);  mul_140 = None
    sum_134: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 1]);  mul_598 = None
    sum_135: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_421, [0, 1]);  add_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_422: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_408, mul_597);  add_408 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1160: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_422, [128, 4096])
    mm_282: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1160, permute_775);  permute_775 = None
    permute_776: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1160, [1, 0])
    mm_283: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_776, view_392);  view_392 = None
    permute_777: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_136: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True)
    view_1161: "f32[4096]" = torch.ops.aten.reshape.default(sum_136, [4096]);  sum_136 = None
    permute_778: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_1162: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_282, [1, 128, 16384]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_599: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1162, mul_136);  mul_136 = None
    mul_600: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1162, add_109);  view_1162 = add_109 = None
    mul_601: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
    sub_133: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_601);  mul_601 = None
    mul_602: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_599, sub_133);  mul_599 = sub_133 = None
    mul_603: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_602, 0.7978845608028654);  mul_602 = None
    mul_604: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_603, 0.044715)
    pow_43: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_391, 2.0);  view_391 = None
    mul_605: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
    mul_606: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_423: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_603, mul_606);  mul_603 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_607: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_600, 0.5);  mul_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_424: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_423, mul_607);  add_423 = mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1163: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_424, [128, 16384]);  add_424 = None
    mm_284: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1163, permute_779);  permute_779 = None
    permute_780: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1163, [1, 0])
    mm_285: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_780, view_366);  permute_780 = None
    permute_781: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_137: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
    view_1164: "f32[16384]" = torch.ops.aten.reshape.default(sum_137, [16384]);  sum_137 = None
    permute_782: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_1165: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_284, [1, 128, 4096]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_286: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_776, view_388);  permute_776 = view_388 = None
    permute_784: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_286, [1, 0]);  mm_286 = None
    mm_287: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1160, permute_785);  view_1160 = permute_785 = None
    view_1167: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_287, [1, 128, 4096]);  mm_287 = None
    permute_786: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_784, [1, 0]);  permute_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1168: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1167, [1, 128, 16, 256]);  view_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_787: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1168, [0, 2, 1, 3]);  view_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1169: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_787, [16, 128, 256]);  permute_787 = None
    bmm_112: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_788, view_1169);  permute_788 = None
    bmm_113: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1169, permute_789);  view_1169 = permute_789 = None
    view_1170: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_112, [1, 16, 128, 256]);  bmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_425: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_30, view_1170);  tangents_30 = view_1170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1171: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_608: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1171, alias_87);  view_1171 = None
    sum_138: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [-1], True)
    mul_609: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_87, sum_138);  alias_87 = sum_138 = None
    sub_134: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_87: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_134, primals_327);  sub_134 = primals_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_46: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_672, div_87, full_default_29);  slice_672 = div_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1172: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_46, [16, 128, 128]);  where_46 = None
    bmm_114: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_790, view_1172);  permute_790 = None
    bmm_115: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1172, permute_791);  view_1172 = permute_791 = None
    view_1173: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_114, [1, 16, 256, 128]);  bmm_114 = None
    view_1174: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_115, [1, 16, 128, 256]);  bmm_115 = None
    permute_792: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1173, [0, 1, 3, 2]);  view_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_426: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_29, permute_792);  tangents_29 = permute_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_793: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1174, [0, 2, 1, 3]);  view_1174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_794: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_426, [0, 2, 1, 3]);  add_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1404: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_793, 3, 0, 64)
    slice_1405: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_793, 3, 64, 256);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1406: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_794, 3, 0, 64)
    slice_1407: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_794, 3, 64, 256);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_610: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1404, view_375)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1175: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_610, [1, 128, 16, 32, 2]);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_56: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1175, 4, 0)
    select_57: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1175, 4, 1);  view_1175 = None
    neg_85: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_56);  select_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_450: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_85, 3, 1, 9223372036854775807, 2);  neg_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_454: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_427: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_450, slice_scatter_454);  slice_scatter_450 = slice_scatter_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_611: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1404, view_376);  slice_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_428: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_427, mul_611);  add_427 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_612: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1406, view_375);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1176: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_612, [1, 128, 16, 32, 2]);  mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_58: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1176, 4, 0)
    select_59: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1176, 4, 1);  view_1176 = None
    neg_86: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_58);  select_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_458: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_86, 3, 1, 9223372036854775807, 2);  neg_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_462: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_429: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_458, slice_scatter_462);  slice_scatter_458 = slice_scatter_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_613: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1406, view_376);  slice_1406 = view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_430: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_429, mul_613);  add_429 = mul_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_466: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1405, 3, 64, 9223372036854775807);  slice_1405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_470: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_428, 3, 0, 64);  add_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_431: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_466, slice_scatter_470);  slice_scatter_466 = slice_scatter_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_474: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1407, 3, 64, 9223372036854775807);  slice_1407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_478: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_430, 3, 0, 64);  add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_432: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_474, slice_scatter_478);  slice_scatter_474 = slice_scatter_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_795: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_425, [0, 2, 1, 3]);  add_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_239: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
    view_1177: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_239, [1, 128, 4096]);  clone_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1178: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_432, [1, 128, 4096]);  add_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1179: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_431, [1, 128, 4096]);  add_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1180: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1177, [128, 4096]);  view_1177 = None
    permute_796: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1180, [1, 0])
    mm_288: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_796, view_366);  permute_796 = None
    permute_797: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_288, [1, 0]);  mm_288 = None
    mm_289: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1180, permute_798);  view_1180 = permute_798 = None
    view_1181: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_289, [1, 128, 4096]);  mm_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_433: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1165, view_1181);  view_1165 = view_1181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_799: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1182: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1178, [128, 4096]);  view_1178 = None
    permute_800: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1182, [1, 0])
    mm_290: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_800, view_366);  permute_800 = None
    permute_801: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_290, [1, 0]);  mm_290 = None
    mm_291: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1182, permute_802);  view_1182 = permute_802 = None
    view_1183: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_434: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_433, view_1183);  add_433 = view_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_803: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_801, [1, 0]);  permute_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1184: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1179, [128, 4096]);  view_1179 = None
    permute_804: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1184, [1, 0])
    mm_292: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_804, view_366);  permute_804 = view_366 = None
    permute_805: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_292, [1, 0]);  mm_292 = None
    mm_293: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1184, permute_806);  view_1184 = permute_806 = None
    view_1185: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_293, [1, 128, 4096]);  mm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_435: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_434, view_1185);  add_434 = view_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_807: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_615: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_435, primals_132);  primals_132 = None
    mul_616: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_615, 4096)
    sum_139: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True)
    mul_617: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_615, mul_130);  mul_615 = None
    sum_140: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True);  mul_617 = None
    mul_618: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, sum_140);  sum_140 = None
    sub_136: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_616, sum_139);  mul_616 = sum_139 = None
    sub_137: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_136, mul_618);  sub_136 = mul_618 = None
    mul_619: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_88, sub_137);  div_88 = sub_137 = None
    mul_620: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_435, mul_130);  mul_130 = None
    sum_141: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1]);  mul_620 = None
    sum_142: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_435, [0, 1]);  add_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_436: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_422, mul_619);  add_422 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1186: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_436, [128, 4096])
    mm_294: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1186, permute_808);  permute_808 = None
    permute_809: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1186, [1, 0])
    mm_295: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_809, view_364);  view_364 = None
    permute_810: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
    sum_143: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1186, [0], True)
    view_1187: "f32[4096]" = torch.ops.aten.reshape.default(sum_143, [4096]);  sum_143 = None
    permute_811: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    view_1188: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_294, [1, 128, 16384]);  mm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_621: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1188, mul_126);  mul_126 = None
    mul_622: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1188, add_101);  view_1188 = add_101 = None
    mul_623: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
    sub_138: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_623);  mul_623 = None
    mul_624: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_621, sub_138);  mul_621 = sub_138 = None
    mul_625: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_624, 0.7978845608028654);  mul_624 = None
    mul_626: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_625, 0.044715)
    pow_44: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_363, 2.0);  view_363 = None
    mul_627: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
    mul_628: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_626, mul_627);  mul_626 = mul_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_437: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_625, mul_628);  mul_625 = mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_629: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_622, 0.5);  mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_438: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_437, mul_629);  add_437 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1189: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_438, [128, 16384]);  add_438 = None
    mm_296: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1189, permute_812);  permute_812 = None
    permute_813: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1189, [1, 0])
    mm_297: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_813, view_338);  permute_813 = None
    permute_814: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    sum_144: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1189, [0], True);  view_1189 = None
    view_1190: "f32[16384]" = torch.ops.aten.reshape.default(sum_144, [16384]);  sum_144 = None
    permute_815: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_1191: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_296, [1, 128, 4096]);  mm_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_298: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_809, view_360);  permute_809 = view_360 = None
    permute_817: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_298, [1, 0]);  mm_298 = None
    mm_299: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1186, permute_818);  view_1186 = permute_818 = None
    view_1193: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_299, [1, 128, 4096]);  mm_299 = None
    permute_819: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_817, [1, 0]);  permute_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1194: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1193, [1, 128, 16, 256]);  view_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_820: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1195: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_820, [16, 128, 256]);  permute_820 = None
    bmm_116: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_821, view_1195);  permute_821 = None
    bmm_117: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1195, permute_822);  view_1195 = permute_822 = None
    view_1196: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_116, [1, 16, 128, 256]);  bmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_439: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_28, view_1196);  tangents_28 = view_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1197: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_630: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1197, alias_89);  view_1197 = None
    sum_145: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [-1], True)
    mul_631: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_145);  alias_89 = sum_145 = None
    sub_139: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_89: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_139, primals_324);  sub_139 = primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_47: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_624, div_89, full_default_29);  slice_624 = div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1198: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_47, [16, 128, 128]);  where_47 = None
    bmm_118: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_823, view_1198);  permute_823 = None
    bmm_119: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1198, permute_824);  view_1198 = permute_824 = None
    view_1199: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_118, [1, 16, 256, 128]);  bmm_118 = None
    view_1200: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_119, [1, 16, 128, 256]);  bmm_119 = None
    permute_825: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1199, [0, 1, 3, 2]);  view_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_440: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_27, permute_825);  tangents_27 = permute_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_826: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1200, [0, 2, 1, 3]);  view_1200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_827: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_440, [0, 2, 1, 3]);  add_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1408: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_826, 3, 0, 64)
    slice_1409: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_826, 3, 64, 256);  permute_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1410: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_827, 3, 0, 64)
    slice_1411: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_827, 3, 64, 256);  permute_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_632: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1408, view_347)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1201: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_632, [1, 128, 16, 32, 2]);  mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_60: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1201, 4, 0)
    select_61: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1201, 4, 1);  view_1201 = None
    neg_87: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_60);  select_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_482: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_87, 3, 1, 9223372036854775807, 2);  neg_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_486: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_441: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_482, slice_scatter_486);  slice_scatter_482 = slice_scatter_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_633: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1408, view_348);  slice_1408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_442: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_441, mul_633);  add_441 = mul_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_634: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1410, view_347);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1202: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_634, [1, 128, 16, 32, 2]);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_62: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1202, 4, 0)
    select_63: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1202, 4, 1);  view_1202 = None
    neg_88: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_62);  select_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_490: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_88, 3, 1, 9223372036854775807, 2);  neg_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_494: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_443: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_490, slice_scatter_494);  slice_scatter_490 = slice_scatter_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_635: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1410, view_348);  slice_1410 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_444: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_443, mul_635);  add_443 = mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_498: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1409, 3, 64, 9223372036854775807);  slice_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_502: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_442, 3, 0, 64);  add_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_445: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_498, slice_scatter_502);  slice_scatter_498 = slice_scatter_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_506: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1411, 3, 64, 9223372036854775807);  slice_1411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_510: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_444, 3, 0, 64);  add_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_446: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_506, slice_scatter_510);  slice_scatter_506 = slice_scatter_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_828: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_439, [0, 2, 1, 3]);  add_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_240: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
    view_1203: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_240, [1, 128, 4096]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1204: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_446, [1, 128, 4096]);  add_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1205: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_445, [1, 128, 4096]);  add_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1206: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1203, [128, 4096]);  view_1203 = None
    permute_829: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1206, [1, 0])
    mm_300: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_829, view_338);  permute_829 = None
    permute_830: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_300, [1, 0]);  mm_300 = None
    mm_301: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1206, permute_831);  view_1206 = permute_831 = None
    view_1207: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_301, [1, 128, 4096]);  mm_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_447: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1191, view_1207);  view_1191 = view_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_832: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_830, [1, 0]);  permute_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1208: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1204, [128, 4096]);  view_1204 = None
    permute_833: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1208, [1, 0])
    mm_302: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_833, view_338);  permute_833 = None
    permute_834: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_302, [1, 0]);  mm_302 = None
    mm_303: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1208, permute_835);  view_1208 = permute_835 = None
    view_1209: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_448: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_447, view_1209);  add_447 = view_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_836: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1210: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1205, [128, 4096]);  view_1205 = None
    permute_837: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1210, [1, 0])
    mm_304: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_837, view_338);  permute_837 = view_338 = None
    permute_838: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_304, [1, 0]);  mm_304 = None
    mm_305: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1210, permute_839);  view_1210 = permute_839 = None
    view_1211: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_305, [1, 128, 4096]);  mm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_449: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_448, view_1211);  add_448 = view_1211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_840: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_838, [1, 0]);  permute_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_637: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_449, primals_122);  primals_122 = None
    mul_638: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_637, 4096)
    sum_146: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True)
    mul_639: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_637, mul_120);  mul_637 = None
    sum_147: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_639, [2], True);  mul_639 = None
    mul_640: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, sum_147);  sum_147 = None
    sub_141: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_638, sum_146);  mul_638 = sum_146 = None
    sub_142: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_141, mul_640);  sub_141 = mul_640 = None
    mul_641: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_90, sub_142);  div_90 = sub_142 = None
    mul_642: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_449, mul_120);  mul_120 = None
    sum_148: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 1]);  mul_642 = None
    sum_149: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_449, [0, 1]);  add_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_450: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_436, mul_641);  add_436 = mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1212: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_450, [128, 4096])
    mm_306: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1212, permute_841);  permute_841 = None
    permute_842: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1212, [1, 0])
    mm_307: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_842, view_336);  view_336 = None
    permute_843: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
    sum_150: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1212, [0], True)
    view_1213: "f32[4096]" = torch.ops.aten.reshape.default(sum_150, [4096]);  sum_150 = None
    permute_844: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    view_1214: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_306, [1, 128, 16384]);  mm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_643: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1214, mul_116);  mul_116 = None
    mul_644: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1214, add_93);  view_1214 = add_93 = None
    mul_645: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
    sub_143: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_645);  mul_645 = None
    mul_646: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_643, sub_143);  mul_643 = sub_143 = None
    mul_647: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_646, 0.7978845608028654);  mul_646 = None
    mul_648: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_647, 0.044715)
    pow_45: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_335, 2.0);  view_335 = None
    mul_649: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
    mul_650: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_451: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_647, mul_650);  mul_647 = mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_651: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_644, 0.5);  mul_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_452: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_451, mul_651);  add_451 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1215: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_452, [128, 16384]);  add_452 = None
    mm_308: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1215, permute_845);  permute_845 = None
    permute_846: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1215, [1, 0])
    mm_309: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_846, view_310);  permute_846 = None
    permute_847: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    sum_151: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1215, [0], True);  view_1215 = None
    view_1216: "f32[16384]" = torch.ops.aten.reshape.default(sum_151, [16384]);  sum_151 = None
    permute_848: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    view_1217: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_308, [1, 128, 4096]);  mm_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_310: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_842, view_332);  permute_842 = view_332 = None
    permute_850: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_310, [1, 0]);  mm_310 = None
    mm_311: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1212, permute_851);  view_1212 = permute_851 = None
    view_1219: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_311, [1, 128, 4096]);  mm_311 = None
    permute_852: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_850, [1, 0]);  permute_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1220: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1219, [1, 128, 16, 256]);  view_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_853: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1220, [0, 2, 1, 3]);  view_1220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1221: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_853, [16, 128, 256]);  permute_853 = None
    bmm_120: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_854, view_1221);  permute_854 = None
    bmm_121: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1221, permute_855);  view_1221 = permute_855 = None
    view_1222: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_120, [1, 16, 128, 256]);  bmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_453: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_26, view_1222);  tangents_26 = view_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1223: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_652: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1223, alias_91);  view_1223 = None
    sum_152: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [-1], True)
    mul_653: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_91, sum_152);  alias_91 = sum_152 = None
    sub_144: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_91: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_144, primals_321);  sub_144 = primals_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_48: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_576, div_91, full_default_29);  slice_576 = div_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1224: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_48, [16, 128, 128]);  where_48 = None
    bmm_122: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_856, view_1224);  permute_856 = None
    bmm_123: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1224, permute_857);  view_1224 = permute_857 = None
    view_1225: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_122, [1, 16, 256, 128]);  bmm_122 = None
    view_1226: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_123, [1, 16, 128, 256]);  bmm_123 = None
    permute_858: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1225, [0, 1, 3, 2]);  view_1225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_454: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_25, permute_858);  tangents_25 = permute_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_859: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1226, [0, 2, 1, 3]);  view_1226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_860: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_454, [0, 2, 1, 3]);  add_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1412: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_859, 3, 0, 64)
    slice_1413: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_859, 3, 64, 256);  permute_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1414: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_860, 3, 0, 64)
    slice_1415: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_860, 3, 64, 256);  permute_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_654: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1412, view_319)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1227: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_654, [1, 128, 16, 32, 2]);  mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_64: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1227, 4, 0)
    select_65: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1227, 4, 1);  view_1227 = None
    neg_89: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_64);  select_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_514: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_89, 3, 1, 9223372036854775807, 2);  neg_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_518: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_455: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_514, slice_scatter_518);  slice_scatter_514 = slice_scatter_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_655: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1412, view_320);  slice_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_456: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_455, mul_655);  add_455 = mul_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_656: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1414, view_319);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1228: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_656, [1, 128, 16, 32, 2]);  mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_66: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1228, 4, 0)
    select_67: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1228, 4, 1);  view_1228 = None
    neg_90: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_66);  select_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_522: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_90, 3, 1, 9223372036854775807, 2);  neg_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_526: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_457: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_522, slice_scatter_526);  slice_scatter_522 = slice_scatter_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_657: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1414, view_320);  slice_1414 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_458: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_457, mul_657);  add_457 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_530: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1413, 3, 64, 9223372036854775807);  slice_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_534: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_456, 3, 0, 64);  add_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_459: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_530, slice_scatter_534);  slice_scatter_530 = slice_scatter_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_538: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1415, 3, 64, 9223372036854775807);  slice_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_542: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_458, 3, 0, 64);  add_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_460: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_538, slice_scatter_542);  slice_scatter_538 = slice_scatter_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_861: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_453, [0, 2, 1, 3]);  add_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_241: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
    view_1229: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_241, [1, 128, 4096]);  clone_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1230: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_460, [1, 128, 4096]);  add_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1231: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_459, [1, 128, 4096]);  add_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1232: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1229, [128, 4096]);  view_1229 = None
    permute_862: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1232, [1, 0])
    mm_312: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_862, view_310);  permute_862 = None
    permute_863: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_312, [1, 0]);  mm_312 = None
    mm_313: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1232, permute_864);  view_1232 = permute_864 = None
    view_1233: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_313, [1, 128, 4096]);  mm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_461: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1217, view_1233);  view_1217 = view_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_865: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1234: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1230, [128, 4096]);  view_1230 = None
    permute_866: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1234, [1, 0])
    mm_314: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_866, view_310);  permute_866 = None
    permute_867: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_314, [1, 0]);  mm_314 = None
    mm_315: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1234, permute_868);  view_1234 = permute_868 = None
    view_1235: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_462: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_461, view_1235);  add_461 = view_1235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_869: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1236: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1231, [128, 4096]);  view_1231 = None
    permute_870: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1236, [1, 0])
    mm_316: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_870, view_310);  permute_870 = view_310 = None
    permute_871: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_316, [1, 0]);  mm_316 = None
    mm_317: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1236, permute_872);  view_1236 = permute_872 = None
    view_1237: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_317, [1, 128, 4096]);  mm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_463: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_462, view_1237);  add_462 = view_1237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_873: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_659: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_463, primals_112);  primals_112 = None
    mul_660: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_659, 4096)
    sum_153: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True)
    mul_661: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_659, mul_110);  mul_659 = None
    sum_154: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True);  mul_661 = None
    mul_662: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, sum_154);  sum_154 = None
    sub_146: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_660, sum_153);  mul_660 = sum_153 = None
    sub_147: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_146, mul_662);  sub_146 = mul_662 = None
    mul_663: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_92, sub_147);  div_92 = sub_147 = None
    mul_664: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_463, mul_110);  mul_110 = None
    sum_155: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_664, [0, 1]);  mul_664 = None
    sum_156: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_463, [0, 1]);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_464: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_450, mul_663);  add_450 = mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1238: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_464, [128, 4096])
    mm_318: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1238, permute_874);  permute_874 = None
    permute_875: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1238, [1, 0])
    mm_319: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_875, view_308);  view_308 = None
    permute_876: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_319, [1, 0]);  mm_319 = None
    sum_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1238, [0], True)
    view_1239: "f32[4096]" = torch.ops.aten.reshape.default(sum_157, [4096]);  sum_157 = None
    permute_877: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    view_1240: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_318, [1, 128, 16384]);  mm_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_665: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1240, mul_106);  mul_106 = None
    mul_666: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1240, add_85);  view_1240 = add_85 = None
    mul_667: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
    sub_148: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_667);  mul_667 = None
    mul_668: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_665, sub_148);  mul_665 = sub_148 = None
    mul_669: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_668, 0.7978845608028654);  mul_668 = None
    mul_670: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_669, 0.044715)
    pow_46: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 2.0);  view_307 = None
    mul_671: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
    mul_672: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_465: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_669, mul_672);  mul_669 = mul_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_673: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_666, 0.5);  mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_466: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_465, mul_673);  add_465 = mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1241: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_466, [128, 16384]);  add_466 = None
    mm_320: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1241, permute_878);  permute_878 = None
    permute_879: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1241, [1, 0])
    mm_321: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_879, view_282);  permute_879 = None
    permute_880: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
    sum_158: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1241, [0], True);  view_1241 = None
    view_1242: "f32[16384]" = torch.ops.aten.reshape.default(sum_158, [16384]);  sum_158 = None
    permute_881: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_880, [1, 0]);  permute_880 = None
    view_1243: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_320, [1, 128, 4096]);  mm_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_322: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_875, view_304);  permute_875 = view_304 = None
    permute_883: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_322, [1, 0]);  mm_322 = None
    mm_323: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1238, permute_884);  view_1238 = permute_884 = None
    view_1245: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_323, [1, 128, 4096]);  mm_323 = None
    permute_885: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_883, [1, 0]);  permute_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1246: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1245, [1, 128, 16, 256]);  view_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_886: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1246, [0, 2, 1, 3]);  view_1246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1247: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_886, [16, 128, 256]);  permute_886 = None
    bmm_124: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_887, view_1247);  permute_887 = None
    bmm_125: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1247, permute_888);  view_1247 = permute_888 = None
    view_1248: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_124, [1, 16, 128, 256]);  bmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_467: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_24, view_1248);  tangents_24 = view_1248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1249: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_674: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1249, alias_93);  view_1249 = None
    sum_159: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [-1], True)
    mul_675: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_93, sum_159);  alias_93 = sum_159 = None
    sub_149: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_93: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_149, primals_318);  sub_149 = primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_49: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_528, div_93, full_default_29);  slice_528 = div_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1250: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_49, [16, 128, 128]);  where_49 = None
    bmm_126: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_889, view_1250);  permute_889 = None
    bmm_127: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1250, permute_890);  view_1250 = permute_890 = None
    view_1251: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_126, [1, 16, 256, 128]);  bmm_126 = None
    view_1252: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_127, [1, 16, 128, 256]);  bmm_127 = None
    permute_891: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1251, [0, 1, 3, 2]);  view_1251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_468: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_23, permute_891);  tangents_23 = permute_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_892: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1252, [0, 2, 1, 3]);  view_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_893: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_468, [0, 2, 1, 3]);  add_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1416: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_892, 3, 0, 64)
    slice_1417: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_892, 3, 64, 256);  permute_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1418: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_893, 3, 0, 64)
    slice_1419: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_893, 3, 64, 256);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_676: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1416, view_291)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1253: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_676, [1, 128, 16, 32, 2]);  mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_68: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1253, 4, 0)
    select_69: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1253, 4, 1);  view_1253 = None
    neg_91: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_68);  select_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_546: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_91, 3, 1, 9223372036854775807, 2);  neg_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_550: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_469: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_546, slice_scatter_550);  slice_scatter_546 = slice_scatter_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_677: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1416, view_292);  slice_1416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_470: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_469, mul_677);  add_469 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_678: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1418, view_291);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1254: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_678, [1, 128, 16, 32, 2]);  mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_70: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1254, 4, 0)
    select_71: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1254, 4, 1);  view_1254 = None
    neg_92: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_70);  select_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_554: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_92, 3, 1, 9223372036854775807, 2);  neg_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_558: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_471: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_554, slice_scatter_558);  slice_scatter_554 = slice_scatter_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_679: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1418, view_292);  slice_1418 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_472: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_471, mul_679);  add_471 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_562: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1417, 3, 64, 9223372036854775807);  slice_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_566: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_470, 3, 0, 64);  add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_473: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_562, slice_scatter_566);  slice_scatter_562 = slice_scatter_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_570: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1419, 3, 64, 9223372036854775807);  slice_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_574: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_472, 3, 0, 64);  add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_474: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_570, slice_scatter_574);  slice_scatter_570 = slice_scatter_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_894: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_467, [0, 2, 1, 3]);  add_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_242: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
    view_1255: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_242, [1, 128, 4096]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1256: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_474, [1, 128, 4096]);  add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1257: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_473, [1, 128, 4096]);  add_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1258: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1255, [128, 4096]);  view_1255 = None
    permute_895: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1258, [1, 0])
    mm_324: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_895, view_282);  permute_895 = None
    permute_896: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_324, [1, 0]);  mm_324 = None
    mm_325: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1258, permute_897);  view_1258 = permute_897 = None
    view_1259: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_325, [1, 128, 4096]);  mm_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_475: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1243, view_1259);  view_1243 = view_1259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_898: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_896, [1, 0]);  permute_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1260: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1256, [128, 4096]);  view_1256 = None
    permute_899: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1260, [1, 0])
    mm_326: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_899, view_282);  permute_899 = None
    permute_900: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_326, [1, 0]);  mm_326 = None
    mm_327: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1260, permute_901);  view_1260 = permute_901 = None
    view_1261: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_476: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_475, view_1261);  add_475 = view_1261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_902: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_900, [1, 0]);  permute_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1262: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1257, [128, 4096]);  view_1257 = None
    permute_903: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1262, [1, 0])
    mm_328: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_903, view_282);  permute_903 = view_282 = None
    permute_904: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_328, [1, 0]);  mm_328 = None
    mm_329: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1262, permute_905);  view_1262 = permute_905 = None
    view_1263: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_329, [1, 128, 4096]);  mm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_477: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_476, view_1263);  add_476 = view_1263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_906: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_904, [1, 0]);  permute_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_681: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_477, primals_102);  primals_102 = None
    mul_682: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_681, 4096)
    sum_160: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_681, [2], True)
    mul_683: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_681, mul_100);  mul_681 = None
    sum_161: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [2], True);  mul_683 = None
    mul_684: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, sum_161);  sum_161 = None
    sub_151: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_682, sum_160);  mul_682 = sum_160 = None
    sub_152: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_151, mul_684);  sub_151 = mul_684 = None
    mul_685: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_94, sub_152);  div_94 = sub_152 = None
    mul_686: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_477, mul_100);  mul_100 = None
    sum_162: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 1]);  mul_686 = None
    sum_163: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_477, [0, 1]);  add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_478: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_464, mul_685);  add_464 = mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1264: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_478, [128, 4096])
    mm_330: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1264, permute_907);  permute_907 = None
    permute_908: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1264, [1, 0])
    mm_331: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_908, view_280);  view_280 = None
    permute_909: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
    sum_164: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1264, [0], True)
    view_1265: "f32[4096]" = torch.ops.aten.reshape.default(sum_164, [4096]);  sum_164 = None
    permute_910: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_909, [1, 0]);  permute_909 = None
    view_1266: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_330, [1, 128, 16384]);  mm_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_687: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1266, mul_96);  mul_96 = None
    mul_688: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1266, add_77);  view_1266 = add_77 = None
    mul_689: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
    sub_153: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_689);  mul_689 = None
    mul_690: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_687, sub_153);  mul_687 = sub_153 = None
    mul_691: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_690, 0.7978845608028654);  mul_690 = None
    mul_692: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_691, 0.044715)
    pow_47: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_279, 2.0);  view_279 = None
    mul_693: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
    mul_694: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_479: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_691, mul_694);  mul_691 = mul_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_695: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_688, 0.5);  mul_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_480: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_479, mul_695);  add_479 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1267: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_480, [128, 16384]);  add_480 = None
    mm_332: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1267, permute_911);  permute_911 = None
    permute_912: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1267, [1, 0])
    mm_333: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_912, view_254);  permute_912 = None
    permute_913: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
    sum_165: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1267, [0], True);  view_1267 = None
    view_1268: "f32[16384]" = torch.ops.aten.reshape.default(sum_165, [16384]);  sum_165 = None
    permute_914: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
    view_1269: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_332, [1, 128, 4096]);  mm_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_334: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_908, view_276);  permute_908 = view_276 = None
    permute_916: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_334, [1, 0]);  mm_334 = None
    mm_335: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1264, permute_917);  view_1264 = permute_917 = None
    view_1271: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_335, [1, 128, 4096]);  mm_335 = None
    permute_918: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_916, [1, 0]);  permute_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1272: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1271, [1, 128, 16, 256]);  view_1271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_919: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1272, [0, 2, 1, 3]);  view_1272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1273: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_919, [16, 128, 256]);  permute_919 = None
    bmm_128: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_920, view_1273);  permute_920 = None
    bmm_129: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1273, permute_921);  view_1273 = permute_921 = None
    view_1274: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_128, [1, 16, 128, 256]);  bmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_481: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_22, view_1274);  tangents_22 = view_1274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1275: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_696: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1275, alias_95);  view_1275 = None
    sum_166: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [-1], True)
    mul_697: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_166);  alias_95 = sum_166 = None
    sub_154: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_95: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_154, primals_315);  sub_154 = primals_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_50: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_480, div_95, full_default_29);  slice_480 = div_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1276: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_50, [16, 128, 128]);  where_50 = None
    bmm_130: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_922, view_1276);  permute_922 = None
    bmm_131: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1276, permute_923);  view_1276 = permute_923 = None
    view_1277: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_130, [1, 16, 256, 128]);  bmm_130 = None
    view_1278: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_131, [1, 16, 128, 256]);  bmm_131 = None
    permute_924: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1277, [0, 1, 3, 2]);  view_1277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_482: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_21, permute_924);  tangents_21 = permute_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_925: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1278, [0, 2, 1, 3]);  view_1278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_926: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_482, [0, 2, 1, 3]);  add_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1420: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_925, 3, 0, 64)
    slice_1421: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_925, 3, 64, 256);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1422: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_926, 3, 0, 64)
    slice_1423: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_926, 3, 64, 256);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_698: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1420, view_263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1279: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_698, [1, 128, 16, 32, 2]);  mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_72: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1279, 4, 0)
    select_73: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1279, 4, 1);  view_1279 = None
    neg_93: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_72);  select_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_578: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_93, 3, 1, 9223372036854775807, 2);  neg_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_582: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_483: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_578, slice_scatter_582);  slice_scatter_578 = slice_scatter_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_699: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1420, view_264);  slice_1420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_484: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_483, mul_699);  add_483 = mul_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_700: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1422, view_263);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1280: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_700, [1, 128, 16, 32, 2]);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_74: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1280, 4, 0)
    select_75: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1280, 4, 1);  view_1280 = None
    neg_94: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_74);  select_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_586: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_94, 3, 1, 9223372036854775807, 2);  neg_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_590: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_485: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_586, slice_scatter_590);  slice_scatter_586 = slice_scatter_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_701: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1422, view_264);  slice_1422 = view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_486: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_485, mul_701);  add_485 = mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_594: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1421, 3, 64, 9223372036854775807);  slice_1421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_598: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_484, 3, 0, 64);  add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_487: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_594, slice_scatter_598);  slice_scatter_594 = slice_scatter_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_602: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1423, 3, 64, 9223372036854775807);  slice_1423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_606: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_486, 3, 0, 64);  add_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_488: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_602, slice_scatter_606);  slice_scatter_602 = slice_scatter_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_927: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_481, [0, 2, 1, 3]);  add_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_243: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_927, memory_format = torch.contiguous_format);  permute_927 = None
    view_1281: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_243, [1, 128, 4096]);  clone_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1282: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_488, [1, 128, 4096]);  add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1283: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_487, [1, 128, 4096]);  add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1284: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1281, [128, 4096]);  view_1281 = None
    permute_928: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1284, [1, 0])
    mm_336: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_928, view_254);  permute_928 = None
    permute_929: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_336, [1, 0]);  mm_336 = None
    mm_337: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1284, permute_930);  view_1284 = permute_930 = None
    view_1285: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_337, [1, 128, 4096]);  mm_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_489: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1269, view_1285);  view_1269 = view_1285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_931: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_929, [1, 0]);  permute_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1286: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1282, [128, 4096]);  view_1282 = None
    permute_932: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1286, [1, 0])
    mm_338: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_932, view_254);  permute_932 = None
    permute_933: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_338, [1, 0]);  mm_338 = None
    mm_339: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1286, permute_934);  view_1286 = permute_934 = None
    view_1287: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_490: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_489, view_1287);  add_489 = view_1287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_935: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_933, [1, 0]);  permute_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1288: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1283, [128, 4096]);  view_1283 = None
    permute_936: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1288, [1, 0])
    mm_340: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_936, view_254);  permute_936 = view_254 = None
    permute_937: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_340, [1, 0]);  mm_340 = None
    mm_341: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1288, permute_938);  view_1288 = permute_938 = None
    view_1289: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_341, [1, 128, 4096]);  mm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_491: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_490, view_1289);  add_490 = view_1289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_939: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_937, [1, 0]);  permute_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_703: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_491, primals_92);  primals_92 = None
    mul_704: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_703, 4096)
    sum_167: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
    mul_705: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_703, mul_90);  mul_703 = None
    sum_168: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
    mul_706: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, sum_168);  sum_168 = None
    sub_156: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_704, sum_167);  mul_704 = sum_167 = None
    sub_157: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_156, mul_706);  sub_156 = mul_706 = None
    mul_707: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_96, sub_157);  div_96 = sub_157 = None
    mul_708: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_491, mul_90);  mul_90 = None
    sum_169: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
    sum_170: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_491, [0, 1]);  add_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_492: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_478, mul_707);  add_478 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1290: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_492, [128, 4096])
    mm_342: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1290, permute_940);  permute_940 = None
    permute_941: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1290, [1, 0])
    mm_343: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_941, view_252);  view_252 = None
    permute_942: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_343, [1, 0]);  mm_343 = None
    sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1290, [0], True)
    view_1291: "f32[4096]" = torch.ops.aten.reshape.default(sum_171, [4096]);  sum_171 = None
    permute_943: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_942, [1, 0]);  permute_942 = None
    view_1292: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_342, [1, 128, 16384]);  mm_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_709: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1292, mul_86);  mul_86 = None
    mul_710: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1292, add_69);  view_1292 = add_69 = None
    mul_711: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
    sub_158: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_711);  mul_711 = None
    mul_712: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_709, sub_158);  mul_709 = sub_158 = None
    mul_713: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_712, 0.7978845608028654);  mul_712 = None
    mul_714: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_713, 0.044715)
    pow_48: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 2.0);  view_251 = None
    mul_715: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
    mul_716: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_493: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_713, mul_716);  mul_713 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_717: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_710, 0.5);  mul_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_494: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_493, mul_717);  add_493 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1293: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_494, [128, 16384]);  add_494 = None
    mm_344: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1293, permute_944);  permute_944 = None
    permute_945: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1293, [1, 0])
    mm_345: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_945, view_226);  permute_945 = None
    permute_946: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
    sum_172: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1293, [0], True);  view_1293 = None
    view_1294: "f32[16384]" = torch.ops.aten.reshape.default(sum_172, [16384]);  sum_172 = None
    permute_947: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_946, [1, 0]);  permute_946 = None
    view_1295: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_344, [1, 128, 4096]);  mm_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_346: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_941, view_248);  permute_941 = view_248 = None
    permute_949: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_346, [1, 0]);  mm_346 = None
    mm_347: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1290, permute_950);  view_1290 = permute_950 = None
    view_1297: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_347, [1, 128, 4096]);  mm_347 = None
    permute_951: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_949, [1, 0]);  permute_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1298: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1297, [1, 128, 16, 256]);  view_1297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_952: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1298, [0, 2, 1, 3]);  view_1298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1299: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_952, [16, 128, 256]);  permute_952 = None
    bmm_132: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_953, view_1299);  permute_953 = None
    bmm_133: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1299, permute_954);  view_1299 = permute_954 = None
    view_1300: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_132, [1, 16, 128, 256]);  bmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_495: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_20, view_1300);  tangents_20 = view_1300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1301: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_718: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1301, alias_97);  view_1301 = None
    sum_173: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [-1], True)
    mul_719: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_97, sum_173);  alias_97 = sum_173 = None
    sub_159: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_97: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_159, primals_312);  sub_159 = primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_51: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_432, div_97, full_default_29);  slice_432 = div_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1302: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_51, [16, 128, 128]);  where_51 = None
    bmm_134: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_955, view_1302);  permute_955 = None
    bmm_135: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1302, permute_956);  view_1302 = permute_956 = None
    view_1303: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_134, [1, 16, 256, 128]);  bmm_134 = None
    view_1304: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_135, [1, 16, 128, 256]);  bmm_135 = None
    permute_957: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1303, [0, 1, 3, 2]);  view_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_496: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_19, permute_957);  tangents_19 = permute_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_958: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1304, [0, 2, 1, 3]);  view_1304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_959: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_496, [0, 2, 1, 3]);  add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1424: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_958, 3, 0, 64)
    slice_1425: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_958, 3, 64, 256);  permute_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1426: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_959, 3, 0, 64)
    slice_1427: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_959, 3, 64, 256);  permute_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_720: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1424, view_235)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1305: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_720, [1, 128, 16, 32, 2]);  mul_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_76: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1305, 4, 0)
    select_77: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1305, 4, 1);  view_1305 = None
    neg_95: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_76);  select_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_610: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_95, 3, 1, 9223372036854775807, 2);  neg_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_614: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_497: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_610, slice_scatter_614);  slice_scatter_610 = slice_scatter_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_721: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1424, view_236);  slice_1424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_498: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_497, mul_721);  add_497 = mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_722: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1426, view_235);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1306: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_722, [1, 128, 16, 32, 2]);  mul_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_78: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1306, 4, 0)
    select_79: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1306, 4, 1);  view_1306 = None
    neg_96: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_78);  select_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_618: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_96, 3, 1, 9223372036854775807, 2);  neg_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_622: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_499: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_618, slice_scatter_622);  slice_scatter_618 = slice_scatter_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_723: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1426, view_236);  slice_1426 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_500: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_499, mul_723);  add_499 = mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_626: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1425, 3, 64, 9223372036854775807);  slice_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_630: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_498, 3, 0, 64);  add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_501: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_626, slice_scatter_630);  slice_scatter_626 = slice_scatter_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_634: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1427, 3, 64, 9223372036854775807);  slice_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_638: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_500, 3, 0, 64);  add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_502: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_634, slice_scatter_638);  slice_scatter_634 = slice_scatter_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_960: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_495, [0, 2, 1, 3]);  add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_244: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_960, memory_format = torch.contiguous_format);  permute_960 = None
    view_1307: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_244, [1, 128, 4096]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1308: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_502, [1, 128, 4096]);  add_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1309: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_501, [1, 128, 4096]);  add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1310: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1307, [128, 4096]);  view_1307 = None
    permute_961: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1310, [1, 0])
    mm_348: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_961, view_226);  permute_961 = None
    permute_962: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_348, [1, 0]);  mm_348 = None
    mm_349: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1310, permute_963);  view_1310 = permute_963 = None
    view_1311: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_349, [1, 128, 4096]);  mm_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_503: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1295, view_1311);  view_1295 = view_1311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_964: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_962, [1, 0]);  permute_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1312: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1308, [128, 4096]);  view_1308 = None
    permute_965: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1312, [1, 0])
    mm_350: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_965, view_226);  permute_965 = None
    permute_966: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_350, [1, 0]);  mm_350 = None
    mm_351: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1312, permute_967);  view_1312 = permute_967 = None
    view_1313: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_504: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_503, view_1313);  add_503 = view_1313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_968: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_966, [1, 0]);  permute_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1314: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1309, [128, 4096]);  view_1309 = None
    permute_969: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1314, [1, 0])
    mm_352: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_969, view_226);  permute_969 = view_226 = None
    permute_970: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_352, [1, 0]);  mm_352 = None
    mm_353: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1314, permute_971);  view_1314 = permute_971 = None
    view_1315: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_353, [1, 128, 4096]);  mm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_505: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_504, view_1315);  add_504 = view_1315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_972: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_970, [1, 0]);  permute_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_725: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_505, primals_82);  primals_82 = None
    mul_726: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_725, 4096)
    sum_174: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True)
    mul_727: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_725, mul_80);  mul_725 = None
    sum_175: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True);  mul_727 = None
    mul_728: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, sum_175);  sum_175 = None
    sub_161: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_726, sum_174);  mul_726 = sum_174 = None
    sub_162: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_161, mul_728);  sub_161 = mul_728 = None
    mul_729: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_98, sub_162);  div_98 = sub_162 = None
    mul_730: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_505, mul_80);  mul_80 = None
    sum_176: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 1]);  mul_730 = None
    sum_177: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_505, [0, 1]);  add_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_506: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_492, mul_729);  add_492 = mul_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1316: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_506, [128, 4096])
    mm_354: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1316, permute_973);  permute_973 = None
    permute_974: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1316, [1, 0])
    mm_355: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_974, view_224);  view_224 = None
    permute_975: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_355, [1, 0]);  mm_355 = None
    sum_178: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1316, [0], True)
    view_1317: "f32[4096]" = torch.ops.aten.reshape.default(sum_178, [4096]);  sum_178 = None
    permute_976: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_975, [1, 0]);  permute_975 = None
    view_1318: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_354, [1, 128, 16384]);  mm_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_731: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1318, mul_76);  mul_76 = None
    mul_732: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1318, add_61);  view_1318 = add_61 = None
    mul_733: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
    sub_163: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_733);  mul_733 = None
    mul_734: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_731, sub_163);  mul_731 = sub_163 = None
    mul_735: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_734, 0.7978845608028654);  mul_734 = None
    mul_736: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_735, 0.044715)
    pow_49: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_223, 2.0);  view_223 = None
    mul_737: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
    mul_738: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_507: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_735, mul_738);  mul_735 = mul_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_739: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_732, 0.5);  mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_508: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_507, mul_739);  add_507 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1319: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_508, [128, 16384]);  add_508 = None
    mm_356: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1319, permute_977);  permute_977 = None
    permute_978: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1319, [1, 0])
    mm_357: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_978, view_198);  permute_978 = None
    permute_979: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
    sum_179: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1319, [0], True);  view_1319 = None
    view_1320: "f32[16384]" = torch.ops.aten.reshape.default(sum_179, [16384]);  sum_179 = None
    permute_980: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_979, [1, 0]);  permute_979 = None
    view_1321: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_356, [1, 128, 4096]);  mm_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_358: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_974, view_220);  permute_974 = view_220 = None
    permute_982: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_358, [1, 0]);  mm_358 = None
    mm_359: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1316, permute_983);  view_1316 = permute_983 = None
    view_1323: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_359, [1, 128, 4096]);  mm_359 = None
    permute_984: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_982, [1, 0]);  permute_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1324: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1323, [1, 128, 16, 256]);  view_1323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_985: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1324, [0, 2, 1, 3]);  view_1324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1325: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_985, [16, 128, 256]);  permute_985 = None
    bmm_136: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_986, view_1325);  permute_986 = None
    bmm_137: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1325, permute_987);  view_1325 = permute_987 = None
    view_1326: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_136, [1, 16, 128, 256]);  bmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_509: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_18, view_1326);  tangents_18 = view_1326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1327: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_740: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1327, alias_99);  view_1327 = None
    sum_180: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_740, [-1], True)
    mul_741: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_99, sum_180);  alias_99 = sum_180 = None
    sub_164: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_740, mul_741);  mul_740 = mul_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_99: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_164, primals_309);  sub_164 = primals_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_52: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_384, div_99, full_default_29);  slice_384 = div_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1328: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_52, [16, 128, 128]);  where_52 = None
    bmm_138: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_988, view_1328);  permute_988 = None
    bmm_139: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1328, permute_989);  view_1328 = permute_989 = None
    view_1329: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_138, [1, 16, 256, 128]);  bmm_138 = None
    view_1330: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_139, [1, 16, 128, 256]);  bmm_139 = None
    permute_990: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1329, [0, 1, 3, 2]);  view_1329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_510: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_17, permute_990);  tangents_17 = permute_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_991: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1330, [0, 2, 1, 3]);  view_1330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_992: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_510, [0, 2, 1, 3]);  add_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1428: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_991, 3, 0, 64)
    slice_1429: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_991, 3, 64, 256);  permute_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1430: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_992, 3, 0, 64)
    slice_1431: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_992, 3, 64, 256);  permute_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_742: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1428, view_207)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1331: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_742, [1, 128, 16, 32, 2]);  mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_80: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1331, 4, 0)
    select_81: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1331, 4, 1);  view_1331 = None
    neg_97: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_80);  select_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_642: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_97, 3, 1, 9223372036854775807, 2);  neg_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_646: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_511: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_642, slice_scatter_646);  slice_scatter_642 = slice_scatter_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_743: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1428, view_208);  slice_1428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_512: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_511, mul_743);  add_511 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_744: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1430, view_207);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1332: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_744, [1, 128, 16, 32, 2]);  mul_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_82: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1332, 4, 0)
    select_83: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1332, 4, 1);  view_1332 = None
    neg_98: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_82);  select_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_650: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_98, 3, 1, 9223372036854775807, 2);  neg_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_654: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_513: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_650, slice_scatter_654);  slice_scatter_650 = slice_scatter_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_745: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1430, view_208);  slice_1430 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_514: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_513, mul_745);  add_513 = mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_658: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1429, 3, 64, 9223372036854775807);  slice_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_662: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_512, 3, 0, 64);  add_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_515: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_658, slice_scatter_662);  slice_scatter_658 = slice_scatter_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_666: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1431, 3, 64, 9223372036854775807);  slice_1431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_670: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_514, 3, 0, 64);  add_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_516: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_666, slice_scatter_670);  slice_scatter_666 = slice_scatter_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_993: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_509, [0, 2, 1, 3]);  add_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_245: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_993, memory_format = torch.contiguous_format);  permute_993 = None
    view_1333: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_245, [1, 128, 4096]);  clone_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1334: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_516, [1, 128, 4096]);  add_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1335: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_515, [1, 128, 4096]);  add_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1336: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1333, [128, 4096]);  view_1333 = None
    permute_994: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1336, [1, 0])
    mm_360: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_994, view_198);  permute_994 = None
    permute_995: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_360, [1, 0]);  mm_360 = None
    mm_361: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1336, permute_996);  view_1336 = permute_996 = None
    view_1337: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_361, [1, 128, 4096]);  mm_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_517: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1321, view_1337);  view_1321 = view_1337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_997: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_995, [1, 0]);  permute_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1338: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1334, [128, 4096]);  view_1334 = None
    permute_998: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1338, [1, 0])
    mm_362: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_998, view_198);  permute_998 = None
    permute_999: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_362, [1, 0]);  mm_362 = None
    mm_363: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1338, permute_1000);  view_1338 = permute_1000 = None
    view_1339: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_518: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_517, view_1339);  add_517 = view_1339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1001: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_999, [1, 0]);  permute_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1340: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1335, [128, 4096]);  view_1335 = None
    permute_1002: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1340, [1, 0])
    mm_364: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1002, view_198);  permute_1002 = view_198 = None
    permute_1003: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_364, [1, 0]);  mm_364 = None
    mm_365: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1340, permute_1004);  view_1340 = permute_1004 = None
    view_1341: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_365, [1, 128, 4096]);  mm_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_519: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_518, view_1341);  add_518 = view_1341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1005: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_747: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_519, primals_72);  primals_72 = None
    mul_748: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_747, 4096)
    sum_181: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
    mul_749: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_747, mul_70);  mul_747 = None
    sum_182: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
    mul_750: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, sum_182);  sum_182 = None
    sub_166: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_748, sum_181);  mul_748 = sum_181 = None
    sub_167: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_166, mul_750);  sub_166 = mul_750 = None
    mul_751: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_100, sub_167);  div_100 = sub_167 = None
    mul_752: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_519, mul_70);  mul_70 = None
    sum_183: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
    sum_184: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_519, [0, 1]);  add_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_520: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_506, mul_751);  add_506 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1342: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_520, [128, 4096])
    mm_366: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1342, permute_1006);  permute_1006 = None
    permute_1007: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1342, [1, 0])
    mm_367: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1007, view_196);  view_196 = None
    permute_1008: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_367, [1, 0]);  mm_367 = None
    sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1342, [0], True)
    view_1343: "f32[4096]" = torch.ops.aten.reshape.default(sum_185, [4096]);  sum_185 = None
    permute_1009: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1008, [1, 0]);  permute_1008 = None
    view_1344: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_366, [1, 128, 16384]);  mm_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_753: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1344, mul_66);  mul_66 = None
    mul_754: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1344, add_53);  view_1344 = add_53 = None
    mul_755: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
    sub_168: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_755);  mul_755 = None
    mul_756: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_753, sub_168);  mul_753 = sub_168 = None
    mul_757: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_756, 0.7978845608028654);  mul_756 = None
    mul_758: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_757, 0.044715)
    pow_50: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 2.0);  view_195 = None
    mul_759: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
    mul_760: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_758, mul_759);  mul_758 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_521: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_757, mul_760);  mul_757 = mul_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_761: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_754, 0.5);  mul_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_522: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_521, mul_761);  add_521 = mul_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1345: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_522, [128, 16384]);  add_522 = None
    mm_368: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1345, permute_1010);  permute_1010 = None
    permute_1011: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1345, [1, 0])
    mm_369: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1011, view_170);  permute_1011 = None
    permute_1012: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_369, [1, 0]);  mm_369 = None
    sum_186: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1345, [0], True);  view_1345 = None
    view_1346: "f32[16384]" = torch.ops.aten.reshape.default(sum_186, [16384]);  sum_186 = None
    permute_1013: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1012, [1, 0]);  permute_1012 = None
    view_1347: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_368, [1, 128, 4096]);  mm_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_370: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1007, view_192);  permute_1007 = view_192 = None
    permute_1015: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_370, [1, 0]);  mm_370 = None
    mm_371: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1342, permute_1016);  view_1342 = permute_1016 = None
    view_1349: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_371, [1, 128, 4096]);  mm_371 = None
    permute_1017: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1015, [1, 0]);  permute_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1350: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1349, [1, 128, 16, 256]);  view_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1018: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1350, [0, 2, 1, 3]);  view_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1351: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1018, [16, 128, 256]);  permute_1018 = None
    bmm_140: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1019, view_1351);  permute_1019 = None
    bmm_141: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1351, permute_1020);  view_1351 = permute_1020 = None
    view_1352: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_140, [1, 16, 128, 256]);  bmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_523: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_16, view_1352);  tangents_16 = view_1352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1353: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_762: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1353, alias_101);  view_1353 = None
    sum_187: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_762, [-1], True)
    mul_763: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_101, sum_187);  alias_101 = sum_187 = None
    sub_169: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_101: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_169, primals_306);  sub_169 = primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_53: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_336, div_101, full_default_29);  slice_336 = div_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1354: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_53, [16, 128, 128]);  where_53 = None
    bmm_142: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1021, view_1354);  permute_1021 = None
    bmm_143: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1354, permute_1022);  view_1354 = permute_1022 = None
    view_1355: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_142, [1, 16, 256, 128]);  bmm_142 = None
    view_1356: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_143, [1, 16, 128, 256]);  bmm_143 = None
    permute_1023: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1355, [0, 1, 3, 2]);  view_1355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_524: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_15, permute_1023);  tangents_15 = permute_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1024: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1356, [0, 2, 1, 3]);  view_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1025: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_524, [0, 2, 1, 3]);  add_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1432: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 0, 64)
    slice_1433: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 64, 256);  permute_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1434: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 0, 64)
    slice_1435: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 64, 256);  permute_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_764: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1432, view_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1357: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_764, [1, 128, 16, 32, 2]);  mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_84: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1357, 4, 0)
    select_85: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1357, 4, 1);  view_1357 = None
    neg_99: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_84);  select_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_674: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_99, 3, 1, 9223372036854775807, 2);  neg_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_678: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_525: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_674, slice_scatter_678);  slice_scatter_674 = slice_scatter_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_765: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1432, view_180);  slice_1432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_526: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_525, mul_765);  add_525 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_766: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1434, view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1358: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_766, [1, 128, 16, 32, 2]);  mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_86: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1358, 4, 0)
    select_87: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1358, 4, 1);  view_1358 = None
    neg_100: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_86);  select_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_682: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_100, 3, 1, 9223372036854775807, 2);  neg_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_686: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_527: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_682, slice_scatter_686);  slice_scatter_682 = slice_scatter_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_767: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1434, view_180);  slice_1434 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_528: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_527, mul_767);  add_527 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_690: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1433, 3, 64, 9223372036854775807);  slice_1433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_694: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_526, 3, 0, 64);  add_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_529: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_690, slice_scatter_694);  slice_scatter_690 = slice_scatter_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_698: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1435, 3, 64, 9223372036854775807);  slice_1435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_702: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_528, 3, 0, 64);  add_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_530: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_698, slice_scatter_702);  slice_scatter_698 = slice_scatter_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1026: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_523, [0, 2, 1, 3]);  add_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_246: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
    view_1359: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_246, [1, 128, 4096]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1360: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_530, [1, 128, 4096]);  add_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1361: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_529, [1, 128, 4096]);  add_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1362: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1359, [128, 4096]);  view_1359 = None
    permute_1027: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1362, [1, 0])
    mm_372: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1027, view_170);  permute_1027 = None
    permute_1028: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_372, [1, 0]);  mm_372 = None
    mm_373: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1362, permute_1029);  view_1362 = permute_1029 = None
    view_1363: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_373, [1, 128, 4096]);  mm_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_531: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1347, view_1363);  view_1347 = view_1363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1030: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1364: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1360, [128, 4096]);  view_1360 = None
    permute_1031: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1364, [1, 0])
    mm_374: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1031, view_170);  permute_1031 = None
    permute_1032: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_374, [1, 0]);  mm_374 = None
    mm_375: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1364, permute_1033);  view_1364 = permute_1033 = None
    view_1365: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_532: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_531, view_1365);  add_531 = view_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1034: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1032, [1, 0]);  permute_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1366: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1361, [128, 4096]);  view_1361 = None
    permute_1035: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1366, [1, 0])
    mm_376: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1035, view_170);  permute_1035 = view_170 = None
    permute_1036: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_376, [1, 0]);  mm_376 = None
    mm_377: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1366, permute_1037);  view_1366 = permute_1037 = None
    view_1367: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_377, [1, 128, 4096]);  mm_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_533: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_532, view_1367);  add_532 = view_1367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1038: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1036, [1, 0]);  permute_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_769: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_533, primals_62);  primals_62 = None
    mul_770: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_769, 4096)
    sum_188: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [2], True)
    mul_771: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_769, mul_60);  mul_769 = None
    sum_189: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [2], True);  mul_771 = None
    mul_772: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, sum_189);  sum_189 = None
    sub_171: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_770, sum_188);  mul_770 = sum_188 = None
    sub_172: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_171, mul_772);  sub_171 = mul_772 = None
    mul_773: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_102, sub_172);  div_102 = sub_172 = None
    mul_774: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_533, mul_60);  mul_60 = None
    sum_190: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 1]);  mul_774 = None
    sum_191: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_533, [0, 1]);  add_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_534: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_520, mul_773);  add_520 = mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1368: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_534, [128, 4096])
    mm_378: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1368, permute_1039);  permute_1039 = None
    permute_1040: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1368, [1, 0])
    mm_379: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1040, view_168);  view_168 = None
    permute_1041: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_379, [1, 0]);  mm_379 = None
    sum_192: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1368, [0], True)
    view_1369: "f32[4096]" = torch.ops.aten.reshape.default(sum_192, [4096]);  sum_192 = None
    permute_1042: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1041, [1, 0]);  permute_1041 = None
    view_1370: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_378, [1, 128, 16384]);  mm_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_775: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1370, mul_56);  mul_56 = None
    mul_776: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1370, add_45);  view_1370 = add_45 = None
    mul_777: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
    sub_173: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_777);  mul_777 = None
    mul_778: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_775, sub_173);  mul_775 = sub_173 = None
    mul_779: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_778, 0.7978845608028654);  mul_778 = None
    mul_780: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_779, 0.044715)
    pow_51: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_167, 2.0);  view_167 = None
    mul_781: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
    mul_782: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_535: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_779, mul_782);  mul_779 = mul_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_783: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_776, 0.5);  mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_536: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_535, mul_783);  add_535 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1371: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_536, [128, 16384]);  add_536 = None
    mm_380: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1371, permute_1043);  permute_1043 = None
    permute_1044: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1371, [1, 0])
    mm_381: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1044, view_142);  permute_1044 = None
    permute_1045: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_381, [1, 0]);  mm_381 = None
    sum_193: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1371, [0], True);  view_1371 = None
    view_1372: "f32[16384]" = torch.ops.aten.reshape.default(sum_193, [16384]);  sum_193 = None
    permute_1046: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1045, [1, 0]);  permute_1045 = None
    view_1373: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_380, [1, 128, 4096]);  mm_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_382: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1040, view_164);  permute_1040 = view_164 = None
    permute_1048: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_382, [1, 0]);  mm_382 = None
    mm_383: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1368, permute_1049);  view_1368 = permute_1049 = None
    view_1375: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_383, [1, 128, 4096]);  mm_383 = None
    permute_1050: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1048, [1, 0]);  permute_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1376: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1375, [1, 128, 16, 256]);  view_1375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1051: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1376, [0, 2, 1, 3]);  view_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1377: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1051, [16, 128, 256]);  permute_1051 = None
    bmm_144: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1052, view_1377);  permute_1052 = None
    bmm_145: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1377, permute_1053);  view_1377 = permute_1053 = None
    view_1378: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_144, [1, 16, 128, 256]);  bmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_537: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_14, view_1378);  tangents_14 = view_1378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1379: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_145, [1, 16, 128, 128]);  bmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_784: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1379, alias_103);  view_1379 = None
    sum_194: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_784, [-1], True)
    mul_785: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_103, sum_194);  alias_103 = sum_194 = None
    sub_174: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_103: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_174, primals_303);  sub_174 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_54: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_288, div_103, full_default_29);  slice_288 = div_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1380: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_54, [16, 128, 128]);  where_54 = None
    bmm_146: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1054, view_1380);  permute_1054 = None
    bmm_147: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1380, permute_1055);  view_1380 = permute_1055 = None
    view_1381: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_146, [1, 16, 256, 128]);  bmm_146 = None
    view_1382: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_147, [1, 16, 128, 256]);  bmm_147 = None
    permute_1056: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1381, [0, 1, 3, 2]);  view_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_538: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_13, permute_1056);  tangents_13 = permute_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1057: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1382, [0, 2, 1, 3]);  view_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1058: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_538, [0, 2, 1, 3]);  add_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1436: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 0, 64)
    slice_1437: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 64, 256);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1438: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 0, 64)
    slice_1439: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 64, 256);  permute_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_786: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1436, view_151)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1383: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_786, [1, 128, 16, 32, 2]);  mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_88: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1383, 4, 0)
    select_89: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1383, 4, 1);  view_1383 = None
    neg_101: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_88);  select_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_706: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_101, 3, 1, 9223372036854775807, 2);  neg_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_710: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_539: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_706, slice_scatter_710);  slice_scatter_706 = slice_scatter_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_787: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1436, view_152);  slice_1436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_540: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_539, mul_787);  add_539 = mul_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_788: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1438, view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1384: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_788, [1, 128, 16, 32, 2]);  mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_90: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1384, 4, 0)
    select_91: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1384, 4, 1);  view_1384 = None
    neg_102: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_90);  select_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_714: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_102, 3, 1, 9223372036854775807, 2);  neg_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_718: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_541: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_714, slice_scatter_718);  slice_scatter_714 = slice_scatter_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_789: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1438, view_152);  slice_1438 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_542: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_541, mul_789);  add_541 = mul_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_722: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1437, 3, 64, 9223372036854775807);  slice_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_726: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_540, 3, 0, 64);  add_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_543: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_722, slice_scatter_726);  slice_scatter_722 = slice_scatter_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_730: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1439, 3, 64, 9223372036854775807);  slice_1439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_734: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_542, 3, 0, 64);  add_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_544: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_730, slice_scatter_734);  slice_scatter_730 = slice_scatter_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1059: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_537, [0, 2, 1, 3]);  add_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_247: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
    view_1385: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_247, [1, 128, 4096]);  clone_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1386: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_544, [1, 128, 4096]);  add_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1387: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_543, [1, 128, 4096]);  add_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1388: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1385, [128, 4096]);  view_1385 = None
    permute_1060: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1388, [1, 0])
    mm_384: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1060, view_142);  permute_1060 = None
    permute_1061: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_384, [1, 0]);  mm_384 = None
    mm_385: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1388, permute_1062);  view_1388 = permute_1062 = None
    view_1389: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_385, [1, 128, 4096]);  mm_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_545: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1373, view_1389);  view_1373 = view_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1063: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1061, [1, 0]);  permute_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1390: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1386, [128, 4096]);  view_1386 = None
    permute_1064: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1390, [1, 0])
    mm_386: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1064, view_142);  permute_1064 = None
    permute_1065: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_386, [1, 0]);  mm_386 = None
    mm_387: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1390, permute_1066);  view_1390 = permute_1066 = None
    view_1391: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_387, [1, 128, 4096]);  mm_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_546: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_545, view_1391);  add_545 = view_1391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1067: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1065, [1, 0]);  permute_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1392: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1387, [128, 4096]);  view_1387 = None
    permute_1068: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1392, [1, 0])
    mm_388: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1068, view_142);  permute_1068 = view_142 = None
    permute_1069: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_388, [1, 0]);  mm_388 = None
    mm_389: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1392, permute_1070);  view_1392 = permute_1070 = None
    view_1393: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_389, [1, 128, 4096]);  mm_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_547: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_546, view_1393);  add_546 = view_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1071: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1069, [1, 0]);  permute_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_791: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_547, primals_52);  primals_52 = None
    mul_792: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_791, 4096)
    sum_195: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True)
    mul_793: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_791, mul_50);  mul_791 = None
    sum_196: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_793, [2], True);  mul_793 = None
    mul_794: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, sum_196);  sum_196 = None
    sub_176: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_792, sum_195);  mul_792 = sum_195 = None
    sub_177: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_176, mul_794);  sub_176 = mul_794 = None
    mul_795: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_104, sub_177);  div_104 = sub_177 = None
    mul_796: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_547, mul_50);  mul_50 = None
    sum_197: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 1]);  mul_796 = None
    sum_198: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_547, [0, 1]);  add_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_548: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_534, mul_795);  add_534 = mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1394: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_548, [128, 4096])
    mm_390: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1394, permute_1072);  permute_1072 = None
    permute_1073: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1394, [1, 0])
    mm_391: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1073, view_140);  view_140 = None
    permute_1074: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_391, [1, 0]);  mm_391 = None
    sum_199: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True)
    view_1395: "f32[4096]" = torch.ops.aten.reshape.default(sum_199, [4096]);  sum_199 = None
    permute_1075: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1074, [1, 0]);  permute_1074 = None
    view_1396: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_390, [1, 128, 16384]);  mm_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_797: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1396, mul_46);  mul_46 = None
    mul_798: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1396, add_37);  view_1396 = add_37 = None
    mul_799: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
    sub_178: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_799);  mul_799 = None
    mul_800: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_797, sub_178);  mul_797 = sub_178 = None
    mul_801: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_800, 0.7978845608028654);  mul_800 = None
    mul_802: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_801, 0.044715)
    pow_52: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_139, 2.0);  view_139 = None
    mul_803: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
    mul_804: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_549: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_801, mul_804);  mul_801 = mul_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_805: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_798, 0.5);  mul_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_550: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_549, mul_805);  add_549 = mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1397: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_550, [128, 16384]);  add_550 = None
    mm_392: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1397, permute_1076);  permute_1076 = None
    permute_1077: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1397, [1, 0])
    mm_393: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1077, view_114);  permute_1077 = None
    permute_1078: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_393, [1, 0]);  mm_393 = None
    sum_200: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1397, [0], True);  view_1397 = None
    view_1398: "f32[16384]" = torch.ops.aten.reshape.default(sum_200, [16384]);  sum_200 = None
    permute_1079: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1078, [1, 0]);  permute_1078 = None
    view_1399: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_392, [1, 128, 4096]);  mm_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_394: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1073, view_136);  permute_1073 = view_136 = None
    permute_1081: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_394, [1, 0]);  mm_394 = None
    mm_395: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1394, permute_1082);  view_1394 = permute_1082 = None
    view_1401: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_395, [1, 128, 4096]);  mm_395 = None
    permute_1083: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1081, [1, 0]);  permute_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1402: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1401, [1, 128, 16, 256]);  view_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1084: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1403: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1084, [16, 128, 256]);  permute_1084 = None
    bmm_148: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1085, view_1403);  permute_1085 = None
    bmm_149: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1403, permute_1086);  view_1403 = permute_1086 = None
    view_1404: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_148, [1, 16, 128, 256]);  bmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_551: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_12, view_1404);  tangents_12 = view_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1405: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_149, [1, 16, 128, 128]);  bmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_806: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1405, alias_105);  view_1405 = None
    sum_201: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [-1], True)
    mul_807: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_105, sum_201);  alias_105 = sum_201 = None
    sub_179: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_105: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_179, primals_300);  sub_179 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_55: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_240, div_105, full_default_29);  slice_240 = div_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1406: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_55, [16, 128, 128]);  where_55 = None
    bmm_150: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1087, view_1406);  permute_1087 = None
    bmm_151: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1406, permute_1088);  view_1406 = permute_1088 = None
    view_1407: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_150, [1, 16, 256, 128]);  bmm_150 = None
    view_1408: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_151, [1, 16, 128, 256]);  bmm_151 = None
    permute_1089: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1407, [0, 1, 3, 2]);  view_1407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_552: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_11, permute_1089);  tangents_11 = permute_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1090: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1408, [0, 2, 1, 3]);  view_1408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1091: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_552, [0, 2, 1, 3]);  add_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1440: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 0, 64)
    slice_1441: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 64, 256);  permute_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1442: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 0, 64)
    slice_1443: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 64, 256);  permute_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_808: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1440, view_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1409: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_808, [1, 128, 16, 32, 2]);  mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_92: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1409, 4, 0)
    select_93: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1409, 4, 1);  view_1409 = None
    neg_103: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_92);  select_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_738: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_103, 3, 1, 9223372036854775807, 2);  neg_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_742: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_553: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_738, slice_scatter_742);  slice_scatter_738 = slice_scatter_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_809: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1440, view_124);  slice_1440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_554: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_553, mul_809);  add_553 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_810: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1442, view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1410: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_810, [1, 128, 16, 32, 2]);  mul_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_94: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1410, 4, 0)
    select_95: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1410, 4, 1);  view_1410 = None
    neg_104: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_94);  select_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_746: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_104, 3, 1, 9223372036854775807, 2);  neg_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_750: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_555: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_746, slice_scatter_750);  slice_scatter_746 = slice_scatter_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_811: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1442, view_124);  slice_1442 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_556: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_555, mul_811);  add_555 = mul_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_754: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1441, 3, 64, 9223372036854775807);  slice_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_758: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_554, 3, 0, 64);  add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_557: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_754, slice_scatter_758);  slice_scatter_754 = slice_scatter_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_762: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1443, 3, 64, 9223372036854775807);  slice_1443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_766: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_556, 3, 0, 64);  add_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_558: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_762, slice_scatter_766);  slice_scatter_762 = slice_scatter_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1092: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_551, [0, 2, 1, 3]);  add_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_248: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
    view_1411: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_248, [1, 128, 4096]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1412: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_558, [1, 128, 4096]);  add_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1413: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_557, [1, 128, 4096]);  add_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1414: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1411, [128, 4096]);  view_1411 = None
    permute_1093: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1414, [1, 0])
    mm_396: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1093, view_114);  permute_1093 = None
    permute_1094: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_396, [1, 0]);  mm_396 = None
    mm_397: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1414, permute_1095);  view_1414 = permute_1095 = None
    view_1415: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_397, [1, 128, 4096]);  mm_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_559: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1399, view_1415);  view_1399 = view_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1096: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1094, [1, 0]);  permute_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1416: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1412, [128, 4096]);  view_1412 = None
    permute_1097: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1416, [1, 0])
    mm_398: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1097, view_114);  permute_1097 = None
    permute_1098: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_398, [1, 0]);  mm_398 = None
    mm_399: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1416, permute_1099);  view_1416 = permute_1099 = None
    view_1417: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_399, [1, 128, 4096]);  mm_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_560: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_559, view_1417);  add_559 = view_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1100: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1098, [1, 0]);  permute_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1418: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1413, [128, 4096]);  view_1413 = None
    permute_1101: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1418, [1, 0])
    mm_400: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1101, view_114);  permute_1101 = view_114 = None
    permute_1102: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_400, [1, 0]);  mm_400 = None
    mm_401: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1418, permute_1103);  view_1418 = permute_1103 = None
    view_1419: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_401, [1, 128, 4096]);  mm_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_561: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_560, view_1419);  add_560 = view_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1104: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1102, [1, 0]);  permute_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_813: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_561, primals_42);  primals_42 = None
    mul_814: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_813, 4096)
    sum_202: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_813, [2], True)
    mul_815: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_813, mul_40);  mul_813 = None
    sum_203: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True);  mul_815 = None
    mul_816: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, sum_203);  sum_203 = None
    sub_181: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_814, sum_202);  mul_814 = sum_202 = None
    sub_182: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_181, mul_816);  sub_181 = mul_816 = None
    mul_817: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_106, sub_182);  div_106 = sub_182 = None
    mul_818: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_561, mul_40);  mul_40 = None
    sum_204: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1]);  mul_818 = None
    sum_205: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_561, [0, 1]);  add_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_562: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_548, mul_817);  add_548 = mul_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1420: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_562, [128, 4096])
    mm_402: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1420, permute_1105);  permute_1105 = None
    permute_1106: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1420, [1, 0])
    mm_403: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1106, view_112);  view_112 = None
    permute_1107: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_403, [1, 0]);  mm_403 = None
    sum_206: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1420, [0], True)
    view_1421: "f32[4096]" = torch.ops.aten.reshape.default(sum_206, [4096]);  sum_206 = None
    permute_1108: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1107, [1, 0]);  permute_1107 = None
    view_1422: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_402, [1, 128, 16384]);  mm_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_819: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1422, mul_36);  mul_36 = None
    mul_820: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1422, add_29);  view_1422 = add_29 = None
    mul_821: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
    sub_183: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_821);  mul_821 = None
    mul_822: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_819, sub_183);  mul_819 = sub_183 = None
    mul_823: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_822, 0.7978845608028654);  mul_822 = None
    mul_824: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_823, 0.044715)
    pow_53: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_111, 2.0);  view_111 = None
    mul_825: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
    mul_826: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_563: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_823, mul_826);  mul_823 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_827: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_820, 0.5);  mul_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_564: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_563, mul_827);  add_563 = mul_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1423: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_564, [128, 16384]);  add_564 = None
    mm_404: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1423, permute_1109);  permute_1109 = None
    permute_1110: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1423, [1, 0])
    mm_405: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1110, view_86);  permute_1110 = None
    permute_1111: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_405, [1, 0]);  mm_405 = None
    sum_207: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1423, [0], True);  view_1423 = None
    view_1424: "f32[16384]" = torch.ops.aten.reshape.default(sum_207, [16384]);  sum_207 = None
    permute_1112: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1111, [1, 0]);  permute_1111 = None
    view_1425: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_404, [1, 128, 4096]);  mm_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_406: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1106, view_108);  permute_1106 = view_108 = None
    permute_1114: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_406, [1, 0]);  mm_406 = None
    mm_407: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1420, permute_1115);  view_1420 = permute_1115 = None
    view_1427: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_407, [1, 128, 4096]);  mm_407 = None
    permute_1116: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1114, [1, 0]);  permute_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1428: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1427, [1, 128, 16, 256]);  view_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1117: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1428, [0, 2, 1, 3]);  view_1428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1429: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1117, [16, 128, 256]);  permute_1117 = None
    bmm_152: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1118, view_1429);  permute_1118 = None
    bmm_153: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1429, permute_1119);  view_1429 = permute_1119 = None
    view_1430: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_152, [1, 16, 128, 256]);  bmm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_565: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_10, view_1430);  tangents_10 = view_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1431: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_153, [1, 16, 128, 128]);  bmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_828: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1431, alias_107);  view_1431 = None
    sum_208: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_828, [-1], True)
    mul_829: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_107, sum_208);  alias_107 = sum_208 = None
    sub_184: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_107: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_184, primals_297);  sub_184 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_56: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_192, div_107, full_default_29);  slice_192 = div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1432: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_56, [16, 128, 128]);  where_56 = None
    bmm_154: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1120, view_1432);  permute_1120 = None
    bmm_155: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1432, permute_1121);  view_1432 = permute_1121 = None
    view_1433: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_154, [1, 16, 256, 128]);  bmm_154 = None
    view_1434: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_155, [1, 16, 128, 256]);  bmm_155 = None
    permute_1122: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1433, [0, 1, 3, 2]);  view_1433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_566: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_9, permute_1122);  tangents_9 = permute_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1123: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1434, [0, 2, 1, 3]);  view_1434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1124: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_566, [0, 2, 1, 3]);  add_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1444: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 0, 64)
    slice_1445: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 64, 256);  permute_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1446: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 0, 64)
    slice_1447: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 64, 256);  permute_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_830: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1444, view_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1435: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_830, [1, 128, 16, 32, 2]);  mul_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_96: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1435, 4, 0)
    select_97: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1435, 4, 1);  view_1435 = None
    neg_105: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_96);  select_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_770: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_105, 3, 1, 9223372036854775807, 2);  neg_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_774: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_567: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_770, slice_scatter_774);  slice_scatter_770 = slice_scatter_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_831: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1444, view_96);  slice_1444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_568: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_567, mul_831);  add_567 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_832: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1446, view_95);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1436: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_832, [1, 128, 16, 32, 2]);  mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_98: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1436, 4, 0)
    select_99: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1436, 4, 1);  view_1436 = None
    neg_106: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_98);  select_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_778: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_106, 3, 1, 9223372036854775807, 2);  neg_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_782: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_569: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_778, slice_scatter_782);  slice_scatter_778 = slice_scatter_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_833: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1446, view_96);  slice_1446 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_570: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_569, mul_833);  add_569 = mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_786: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1445, 3, 64, 9223372036854775807);  slice_1445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_790: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_568, 3, 0, 64);  add_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_571: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_786, slice_scatter_790);  slice_scatter_786 = slice_scatter_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_794: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1447, 3, 64, 9223372036854775807);  slice_1447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_798: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_570, 3, 0, 64);  add_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_572: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_794, slice_scatter_798);  slice_scatter_794 = slice_scatter_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1125: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_565, [0, 2, 1, 3]);  add_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_249: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
    view_1437: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_249, [1, 128, 4096]);  clone_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1438: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_572, [1, 128, 4096]);  add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1439: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_571, [1, 128, 4096]);  add_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1440: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1437, [128, 4096]);  view_1437 = None
    permute_1126: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1440, [1, 0])
    mm_408: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1126, view_86);  permute_1126 = None
    permute_1127: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_408, [1, 0]);  mm_408 = None
    mm_409: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1440, permute_1128);  view_1440 = permute_1128 = None
    view_1441: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_409, [1, 128, 4096]);  mm_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_573: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1425, view_1441);  view_1425 = view_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1129: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1127, [1, 0]);  permute_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1442: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1438, [128, 4096]);  view_1438 = None
    permute_1130: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1442, [1, 0])
    mm_410: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1130, view_86);  permute_1130 = None
    permute_1131: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_410, [1, 0]);  mm_410 = None
    mm_411: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1442, permute_1132);  view_1442 = permute_1132 = None
    view_1443: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_411, [1, 128, 4096]);  mm_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_574: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_573, view_1443);  add_573 = view_1443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1133: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1131, [1, 0]);  permute_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1444: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1439, [128, 4096]);  view_1439 = None
    permute_1134: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1444, [1, 0])
    mm_412: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1134, view_86);  permute_1134 = view_86 = None
    permute_1135: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_412, [1, 0]);  mm_412 = None
    mm_413: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1444, permute_1136);  view_1444 = permute_1136 = None
    view_1445: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_413, [1, 128, 4096]);  mm_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_575: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_574, view_1445);  add_574 = view_1445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1137: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1135, [1, 0]);  permute_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_835: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_575, primals_32);  primals_32 = None
    mul_836: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_835, 4096)
    sum_209: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True)
    mul_837: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_835, mul_30);  mul_835 = None
    sum_210: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2], True);  mul_837 = None
    mul_838: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, sum_210);  sum_210 = None
    sub_186: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_836, sum_209);  mul_836 = sum_209 = None
    sub_187: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_186, mul_838);  sub_186 = mul_838 = None
    mul_839: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_108, sub_187);  div_108 = sub_187 = None
    mul_840: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_575, mul_30);  mul_30 = None
    sum_211: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 1]);  mul_840 = None
    sum_212: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_575, [0, 1]);  add_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_576: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_562, mul_839);  add_562 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1446: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_576, [128, 4096])
    mm_414: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1446, permute_1138);  permute_1138 = None
    permute_1139: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1446, [1, 0])
    mm_415: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1139, view_84);  view_84 = None
    permute_1140: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_415, [1, 0]);  mm_415 = None
    sum_213: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1446, [0], True)
    view_1447: "f32[4096]" = torch.ops.aten.reshape.default(sum_213, [4096]);  sum_213 = None
    permute_1141: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1140, [1, 0]);  permute_1140 = None
    view_1448: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_414, [1, 128, 16384]);  mm_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_841: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1448, mul_26);  mul_26 = None
    mul_842: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1448, add_21);  view_1448 = add_21 = None
    mul_843: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
    sub_188: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_843);  mul_843 = None
    mul_844: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_841, sub_188);  mul_841 = sub_188 = None
    mul_845: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_844, 0.7978845608028654);  mul_844 = None
    mul_846: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_845, 0.044715)
    pow_54: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_83, 2.0);  view_83 = None
    mul_847: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
    mul_848: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_846, mul_847);  mul_846 = mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_577: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_845, mul_848);  mul_845 = mul_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_849: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_842, 0.5);  mul_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_578: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_577, mul_849);  add_577 = mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1449: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_578, [128, 16384]);  add_578 = None
    mm_416: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1449, permute_1142);  permute_1142 = None
    permute_1143: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1449, [1, 0])
    mm_417: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1143, view_58);  permute_1143 = None
    permute_1144: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_417, [1, 0]);  mm_417 = None
    sum_214: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1449, [0], True);  view_1449 = None
    view_1450: "f32[16384]" = torch.ops.aten.reshape.default(sum_214, [16384]);  sum_214 = None
    permute_1145: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1144, [1, 0]);  permute_1144 = None
    view_1451: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_416, [1, 128, 4096]);  mm_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_418: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1139, view_80);  permute_1139 = view_80 = None
    permute_1147: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_418, [1, 0]);  mm_418 = None
    mm_419: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1446, permute_1148);  view_1446 = permute_1148 = None
    view_1453: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_419, [1, 128, 4096]);  mm_419 = None
    permute_1149: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1147, [1, 0]);  permute_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1454: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1453, [1, 128, 16, 256]);  view_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1150: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1454, [0, 2, 1, 3]);  view_1454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1455: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1150, [16, 128, 256]);  permute_1150 = None
    bmm_156: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1151, view_1455);  permute_1151 = None
    bmm_157: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1455, permute_1152);  view_1455 = permute_1152 = None
    view_1456: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_156, [1, 16, 128, 256]);  bmm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_579: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_8, view_1456);  tangents_8 = view_1456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1457: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_157, [1, 16, 128, 128]);  bmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_850: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1457, alias_109);  view_1457 = None
    sum_215: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_850, [-1], True)
    mul_851: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_109, sum_215);  alias_109 = sum_215 = None
    sub_189: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_850, mul_851);  mul_850 = mul_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_109: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_189, primals_294);  sub_189 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_57: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_144, div_109, full_default_29);  slice_144 = div_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1458: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_57, [16, 128, 128]);  where_57 = None
    bmm_158: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1153, view_1458);  permute_1153 = None
    bmm_159: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1458, permute_1154);  view_1458 = permute_1154 = None
    view_1459: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_158, [1, 16, 256, 128]);  bmm_158 = None
    view_1460: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_159, [1, 16, 128, 256]);  bmm_159 = None
    permute_1155: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1459, [0, 1, 3, 2]);  view_1459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_580: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_7, permute_1155);  tangents_7 = permute_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1156: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1460, [0, 2, 1, 3]);  view_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1157: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_580, [0, 2, 1, 3]);  add_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1448: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 0, 64)
    slice_1449: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 64, 256);  permute_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1450: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 0, 64)
    slice_1451: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 64, 256);  permute_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_852: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1448, view_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1461: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_852, [1, 128, 16, 32, 2]);  mul_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_100: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1461, 4, 0)
    select_101: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1461, 4, 1);  view_1461 = None
    neg_107: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_802: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_107, 3, 1, 9223372036854775807, 2);  neg_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_806: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_581: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_802, slice_scatter_806);  slice_scatter_802 = slice_scatter_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_853: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1448, view_68);  slice_1448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_582: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_581, mul_853);  add_581 = mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_854: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1450, view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1462: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_854, [1, 128, 16, 32, 2]);  mul_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_102: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1462, 4, 0)
    select_103: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1462, 4, 1);  view_1462 = None
    neg_108: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_102);  select_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_810: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_108, 3, 1, 9223372036854775807, 2);  neg_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_814: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_583: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_810, slice_scatter_814);  slice_scatter_810 = slice_scatter_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_855: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1450, view_68);  slice_1450 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_584: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_583, mul_855);  add_583 = mul_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_818: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1449, 3, 64, 9223372036854775807);  slice_1449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_822: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_582, 3, 0, 64);  add_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_585: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_818, slice_scatter_822);  slice_scatter_818 = slice_scatter_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_826: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1451, 3, 64, 9223372036854775807);  slice_1451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_830: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_584, 3, 0, 64);  add_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_586: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_826, slice_scatter_830);  slice_scatter_826 = slice_scatter_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1158: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_579, [0, 2, 1, 3]);  add_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_250: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
    view_1463: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_250, [1, 128, 4096]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1464: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_586, [1, 128, 4096]);  add_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1465: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_585, [1, 128, 4096]);  add_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1466: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1463, [128, 4096]);  view_1463 = None
    permute_1159: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1466, [1, 0])
    mm_420: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1159, view_58);  permute_1159 = None
    permute_1160: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_420, [1, 0]);  mm_420 = None
    mm_421: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1466, permute_1161);  view_1466 = permute_1161 = None
    view_1467: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_421, [1, 128, 4096]);  mm_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_587: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1451, view_1467);  view_1451 = view_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1162: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1160, [1, 0]);  permute_1160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1468: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1464, [128, 4096]);  view_1464 = None
    permute_1163: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1468, [1, 0])
    mm_422: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1163, view_58);  permute_1163 = None
    permute_1164: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_422, [1, 0]);  mm_422 = None
    mm_423: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1468, permute_1165);  view_1468 = permute_1165 = None
    view_1469: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_423, [1, 128, 4096]);  mm_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_588: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_587, view_1469);  add_587 = view_1469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1166: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1164, [1, 0]);  permute_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1470: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1465, [128, 4096]);  view_1465 = None
    permute_1167: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1470, [1, 0])
    mm_424: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1167, view_58);  permute_1167 = view_58 = None
    permute_1168: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_424, [1, 0]);  mm_424 = None
    mm_425: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1470, permute_1169);  view_1470 = permute_1169 = None
    view_1471: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_425, [1, 128, 4096]);  mm_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_589: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_588, view_1471);  add_588 = view_1471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1170: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1168, [1, 0]);  permute_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_857: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_589, primals_22);  primals_22 = None
    mul_858: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_857, 4096)
    sum_216: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_857, [2], True)
    mul_859: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_857, mul_20);  mul_857 = None
    sum_217: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True);  mul_859 = None
    mul_860: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, sum_217);  sum_217 = None
    sub_191: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_858, sum_216);  mul_858 = sum_216 = None
    sub_192: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_191, mul_860);  sub_191 = mul_860 = None
    mul_861: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_110, sub_192);  div_110 = sub_192 = None
    mul_862: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_589, mul_20);  mul_20 = None
    sum_218: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_862, [0, 1]);  mul_862 = None
    sum_219: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_589, [0, 1]);  add_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_590: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_576, mul_861);  add_576 = mul_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1472: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_590, [128, 4096])
    mm_426: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1472, permute_1171);  permute_1171 = None
    permute_1172: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1472, [1, 0])
    mm_427: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1172, view_56);  view_56 = None
    permute_1173: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_427, [1, 0]);  mm_427 = None
    sum_220: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1472, [0], True)
    view_1473: "f32[4096]" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
    permute_1174: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1173, [1, 0]);  permute_1173 = None
    view_1474: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_426, [1, 128, 16384]);  mm_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_863: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1474, mul_16);  mul_16 = None
    mul_864: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1474, add_13);  view_1474 = add_13 = None
    mul_865: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
    sub_193: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_865);  mul_865 = None
    mul_866: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_863, sub_193);  mul_863 = sub_193 = None
    mul_867: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_866, 0.7978845608028654);  mul_866 = None
    mul_868: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_867, 0.044715)
    pow_55: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_55, 2.0);  view_55 = None
    mul_869: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
    mul_870: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_868, mul_869);  mul_868 = mul_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_591: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_867, mul_870);  mul_867 = mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_871: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_864, 0.5);  mul_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_592: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_591, mul_871);  add_591 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1475: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_592, [128, 16384]);  add_592 = None
    mm_428: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1475, permute_1175);  permute_1175 = None
    permute_1176: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1475, [1, 0])
    mm_429: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1176, view_30);  permute_1176 = None
    permute_1177: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_429, [1, 0]);  mm_429 = None
    sum_221: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1475, [0], True);  view_1475 = None
    view_1476: "f32[16384]" = torch.ops.aten.reshape.default(sum_221, [16384]);  sum_221 = None
    permute_1178: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1177, [1, 0]);  permute_1177 = None
    view_1477: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_428, [1, 128, 4096]);  mm_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_430: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1172, view_52);  permute_1172 = view_52 = None
    permute_1180: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_430, [1, 0]);  mm_430 = None
    mm_431: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1472, permute_1181);  view_1472 = permute_1181 = None
    view_1479: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_431, [1, 128, 4096]);  mm_431 = None
    permute_1182: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1180, [1, 0]);  permute_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1480: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1479, [1, 128, 16, 256]);  view_1479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1183: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1480, [0, 2, 1, 3]);  view_1480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1481: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1183, [16, 128, 256]);  permute_1183 = None
    bmm_160: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1184, view_1481);  permute_1184 = None
    bmm_161: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1481, permute_1185);  view_1481 = permute_1185 = None
    view_1482: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_160, [1, 16, 128, 256]);  bmm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_593: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_6, view_1482);  tangents_6 = view_1482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1483: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_161, [1, 16, 128, 128]);  bmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_872: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1483, alias_111);  view_1483 = None
    sum_222: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [-1], True)
    mul_873: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_111, sum_222);  alias_111 = sum_222 = None
    sub_194: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_111: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_194, primals_291);  sub_194 = primals_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_58: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, div_111, full_default_29);  slice_96 = div_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1484: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_58, [16, 128, 128]);  where_58 = None
    bmm_162: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1186, view_1484);  permute_1186 = None
    bmm_163: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1484, permute_1187);  view_1484 = permute_1187 = None
    view_1485: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_162, [1, 16, 256, 128]);  bmm_162 = None
    view_1486: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_163, [1, 16, 128, 256]);  bmm_163 = None
    permute_1188: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1485, [0, 1, 3, 2]);  view_1485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_594: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_5, permute_1188);  tangents_5 = permute_1188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1189: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1486, [0, 2, 1, 3]);  view_1486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1190: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_594, [0, 2, 1, 3]);  add_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1452: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 0, 64)
    slice_1453: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 64, 256);  permute_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1454: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 0, 64)
    slice_1455: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 64, 256);  permute_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_874: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1452, view_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1487: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_874, [1, 128, 16, 32, 2]);  mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_104: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1487, 4, 0)
    select_105: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1487, 4, 1);  view_1487 = None
    neg_109: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_104);  select_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_834: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_109, 3, 1, 9223372036854775807, 2);  neg_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_838: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_595: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_834, slice_scatter_838);  slice_scatter_834 = slice_scatter_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_875: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1452, view_40);  slice_1452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_596: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_595, mul_875);  add_595 = mul_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_876: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1454, view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1488: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_876, [1, 128, 16, 32, 2]);  mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_106: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1488, 4, 0)
    select_107: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1488, 4, 1);  view_1488 = None
    neg_110: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_106);  select_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_842: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_110, 3, 1, 9223372036854775807, 2);  neg_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_846: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_597: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_842, slice_scatter_846);  slice_scatter_842 = slice_scatter_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_877: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1454, view_40);  slice_1454 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_598: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_597, mul_877);  add_597 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_850: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1453, 3, 64, 9223372036854775807);  slice_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_854: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_596, 3, 0, 64);  add_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_599: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_850, slice_scatter_854);  slice_scatter_850 = slice_scatter_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_858: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1455, 3, 64, 9223372036854775807);  slice_1455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_862: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_598, 3, 0, 64);  add_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_600: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_858, slice_scatter_862);  slice_scatter_858 = slice_scatter_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1191: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_593, [0, 2, 1, 3]);  add_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_251: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
    view_1489: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_251, [1, 128, 4096]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1490: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_600, [1, 128, 4096]);  add_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1491: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_599, [1, 128, 4096]);  add_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1492: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1489, [128, 4096]);  view_1489 = None
    permute_1192: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1492, [1, 0])
    mm_432: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1192, view_30);  permute_1192 = None
    permute_1193: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_432, [1, 0]);  mm_432 = None
    mm_433: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1492, permute_1194);  view_1492 = permute_1194 = None
    view_1493: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_433, [1, 128, 4096]);  mm_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_601: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1477, view_1493);  view_1477 = view_1493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1195: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1193, [1, 0]);  permute_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1494: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1490, [128, 4096]);  view_1490 = None
    permute_1196: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1494, [1, 0])
    mm_434: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1196, view_30);  permute_1196 = None
    permute_1197: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_434, [1, 0]);  mm_434 = None
    mm_435: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1494, permute_1198);  view_1494 = permute_1198 = None
    view_1495: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_435, [1, 128, 4096]);  mm_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_602: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_601, view_1495);  add_601 = view_1495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1199: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1197, [1, 0]);  permute_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1496: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1491, [128, 4096]);  view_1491 = None
    permute_1200: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1496, [1, 0])
    mm_436: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1200, view_30);  permute_1200 = view_30 = None
    permute_1201: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_436, [1, 0]);  mm_436 = None
    mm_437: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1496, permute_1202);  view_1496 = permute_1202 = None
    view_1497: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_437, [1, 128, 4096]);  mm_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_603: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_602, view_1497);  add_602 = view_1497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1203: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1201, [1, 0]);  permute_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_879: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_603, primals_12);  primals_12 = None
    mul_880: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_879, 4096)
    sum_223: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [2], True)
    mul_881: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_879, mul_10);  mul_879 = None
    sum_224: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_881, [2], True);  mul_881 = None
    mul_882: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_224);  sum_224 = None
    sub_196: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_880, sum_223);  mul_880 = sum_223 = None
    sub_197: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_196, mul_882);  sub_196 = mul_882 = None
    mul_883: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_112, sub_197);  div_112 = sub_197 = None
    mul_884: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_603, mul_10);  mul_10 = None
    sum_225: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_884, [0, 1]);  mul_884 = None
    sum_226: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_603, [0, 1]);  add_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_604: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_590, mul_883);  add_590 = mul_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1498: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_604, [128, 4096])
    mm_438: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1498, permute_1204);  permute_1204 = None
    permute_1205: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1498, [1, 0])
    mm_439: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1205, view_28);  view_28 = None
    permute_1206: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_439, [1, 0]);  mm_439 = None
    sum_227: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1498, [0], True)
    view_1499: "f32[4096]" = torch.ops.aten.reshape.default(sum_227, [4096]);  sum_227 = None
    permute_1207: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1206, [1, 0]);  permute_1206 = None
    view_1500: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_438, [1, 128, 16384]);  mm_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_885: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1500, mul_6);  mul_6 = None
    mul_886: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1500, add_5);  view_1500 = add_5 = None
    mul_887: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub_198: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_887);  mul_887 = None
    mul_888: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_885, sub_198);  mul_885 = sub_198 = None
    mul_889: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_888, 0.7978845608028654);  mul_888 = None
    mul_890: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_889, 0.044715)
    pow_56: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 2.0);  view_27 = None
    mul_891: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
    mul_892: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_605: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_889, mul_892);  mul_889 = mul_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_893: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_886, 0.5);  mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_606: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_605, mul_893);  add_605 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1501: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_606, [128, 16384]);  add_606 = None
    mm_440: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1501, permute_1208);  permute_1208 = None
    permute_1209: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1501, [1, 0])
    mm_441: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1209, view_2);  permute_1209 = None
    permute_1210: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_441, [1, 0]);  mm_441 = None
    sum_228: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1501, [0], True);  view_1501 = None
    view_1502: "f32[16384]" = torch.ops.aten.reshape.default(sum_228, [16384]);  sum_228 = None
    permute_1211: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1210, [1, 0]);  permute_1210 = None
    view_1503: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_440, [1, 128, 4096]);  mm_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_442: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1205, view_24);  permute_1205 = view_24 = None
    permute_1213: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_442, [1, 0]);  mm_442 = None
    mm_443: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1498, permute_1214);  view_1498 = permute_1214 = None
    view_1505: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_443, [1, 128, 4096]);  mm_443 = None
    permute_1215: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1213, [1, 0]);  permute_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1506: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1505, [1, 128, 16, 256]);  view_1505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1216: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1506, [0, 2, 1, 3]);  view_1506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1507: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1216, [16, 128, 256]);  permute_1216 = None
    bmm_164: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1217, view_1507);  permute_1217 = None
    bmm_165: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1507, permute_1218);  view_1507 = permute_1218 = None
    view_1508: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_164, [1, 16, 128, 256]);  bmm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    add_607: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_4, view_1508);  tangents_4 = view_1508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1509: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_165, [1, 16, 128, 128]);  bmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_894: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1509, alias_113);  view_1509 = None
    sum_229: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [-1], True)
    mul_895: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_113, sum_229);  alias_113 = sum_229 = None
    sub_199: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_113: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_199, primals_288);  sub_199 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_59: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, div_113, full_default_29);  slice_48 = div_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1510: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_59, [16, 128, 128]);  where_59 = None
    bmm_166: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1219, view_1510);  permute_1219 = None
    bmm_167: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1510, permute_1220);  view_1510 = permute_1220 = None
    view_1511: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_166, [1, 16, 256, 128]);  bmm_166 = None
    view_1512: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_167, [1, 16, 128, 256]);  bmm_167 = None
    permute_1221: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1511, [0, 1, 3, 2]);  view_1511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_608: "f32[1, 16, 128, 256]" = torch.ops.aten.add.Tensor(tangents_3, permute_1221);  tangents_3 = permute_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1222: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1512, [0, 2, 1, 3]);  view_1512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1223: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_608, [0, 2, 1, 3]);  add_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1456: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 0, 64)
    slice_1457: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 64, 256);  permute_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1458: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 0, 64)
    slice_1459: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 64, 256);  permute_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_896: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1456, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1513: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_896, [1, 128, 16, 32, 2]);  mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_108: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1513, 4, 0)
    select_109: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1513, 4, 1);  view_1513 = None
    neg_111: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_108);  select_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_866: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_111, 3, 1, 9223372036854775807, 2);  neg_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_870: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_609: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_866, slice_scatter_870);  slice_scatter_866 = slice_scatter_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_897: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1456, view_12);  slice_1456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_610: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_609, mul_897);  add_609 = mul_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_898: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1458, view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1514: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_898, [1, 128, 16, 32, 2]);  mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_110: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1514, 4, 0)
    select_111: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1514, 4, 1);  view_1514 = None
    neg_112: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_110);  select_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_874: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, neg_112, 3, 1, 9223372036854775807, 2);  neg_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_878: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_36, select_111, 3, 0, 9223372036854775807, 2);  full_default_36 = select_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_611: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_874, slice_scatter_878);  slice_scatter_874 = slice_scatter_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_899: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1458, view_12);  slice_1458 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_612: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_611, mul_899);  add_611 = mul_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_882: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1457, 3, 64, 9223372036854775807);  slice_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_886: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_610, 3, 0, 64);  add_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_613: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_882, slice_scatter_886);  slice_scatter_882 = slice_scatter_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_890: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, slice_1459, 3, 64, 9223372036854775807);  slice_1459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_894: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_52, add_612, 3, 0, 64);  full_default_52 = add_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_614: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_890, slice_scatter_894);  slice_scatter_890 = slice_scatter_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1224: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(add_607, [0, 2, 1, 3]);  add_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_252: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
    view_1515: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_252, [1, 128, 4096]);  clone_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1516: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_614, [1, 128, 4096]);  add_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1517: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_613, [1, 128, 4096]);  add_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1518: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1515, [128, 4096]);  view_1515 = None
    permute_1225: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1518, [1, 0])
    mm_444: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1225, view_2);  permute_1225 = None
    permute_1226: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_444, [1, 0]);  mm_444 = None
    mm_445: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1518, permute_1227);  view_1518 = permute_1227 = None
    view_1519: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_445, [1, 128, 4096]);  mm_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_615: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1503, view_1519);  view_1503 = view_1519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1228: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1226, [1, 0]);  permute_1226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1520: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1516, [128, 4096]);  view_1516 = None
    permute_1229: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1520, [1, 0])
    mm_446: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1229, view_2);  permute_1229 = None
    permute_1230: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_446, [1, 0]);  mm_446 = None
    mm_447: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1520, permute_1231);  view_1520 = permute_1231 = None
    view_1521: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_447, [1, 128, 4096]);  mm_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_616: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_615, view_1521);  add_615 = view_1521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1232: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1230, [1, 0]);  permute_1230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1522: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1517, [128, 4096]);  view_1517 = None
    permute_1233: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1522, [1, 0])
    mm_448: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1233, view_2);  permute_1233 = view_2 = None
    permute_1234: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_448, [1, 0]);  mm_448 = None
    mm_449: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1522, permute_1235);  view_1522 = permute_1235 = None
    view_1523: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_449, [1, 128, 4096]);  mm_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_617: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_616, view_1523);  add_616 = view_1523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1236: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1234, [1, 0]);  permute_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_901: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_617, primals_2);  primals_2 = None
    mul_902: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_901, 4096)
    sum_230: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_901, [2], True)
    mul_903: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_901, mul);  mul_901 = None
    sum_231: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_903, [2], True);  mul_903 = None
    mul_904: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, sum_231);  sum_231 = None
    sub_201: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_902, sum_230);  mul_902 = sum_230 = None
    sub_202: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_201, mul_904);  sub_201 = mul_904 = None
    div_114: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
    mul_905: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_114, sub_202);  div_114 = sub_202 = None
    mul_906: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_617, mul);  mul = None
    sum_232: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_906, [0, 1]);  mul_906 = None
    sum_233: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_617, [0, 1]);  add_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_618: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_604, mul_905);  add_604 = mul_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:635, code: inputs_embeds = self.wte(input_ids)
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_367: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_60: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_367, full_default_29, add_618);  unsqueeze_367 = full_default_29 = add_618 = None
    full_default_960: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[50400, 4096]" = torch.ops.prims._unsafe_index_put_.default(full_default_960, [view], where_60, True);  full_default_960 = view = where_60 = None
    return [_unsafe_index_put, sum_232, sum_233, permute_1236, permute_1232, permute_1228, permute_1215, permute_1211, view_1502, permute_1207, view_1499, sum_225, sum_226, permute_1203, permute_1199, permute_1195, permute_1182, permute_1178, view_1476, permute_1174, view_1473, sum_218, sum_219, permute_1170, permute_1166, permute_1162, permute_1149, permute_1145, view_1450, permute_1141, view_1447, sum_211, sum_212, permute_1137, permute_1133, permute_1129, permute_1116, permute_1112, view_1424, permute_1108, view_1421, sum_204, sum_205, permute_1104, permute_1100, permute_1096, permute_1083, permute_1079, view_1398, permute_1075, view_1395, sum_197, sum_198, permute_1071, permute_1067, permute_1063, permute_1050, permute_1046, view_1372, permute_1042, view_1369, sum_190, sum_191, permute_1038, permute_1034, permute_1030, permute_1017, permute_1013, view_1346, permute_1009, view_1343, sum_183, sum_184, permute_1005, permute_1001, permute_997, permute_984, permute_980, view_1320, permute_976, view_1317, sum_176, sum_177, permute_972, permute_968, permute_964, permute_951, permute_947, view_1294, permute_943, view_1291, sum_169, sum_170, permute_939, permute_935, permute_931, permute_918, permute_914, view_1268, permute_910, view_1265, sum_162, sum_163, permute_906, permute_902, permute_898, permute_885, permute_881, view_1242, permute_877, view_1239, sum_155, sum_156, permute_873, permute_869, permute_865, permute_852, permute_848, view_1216, permute_844, view_1213, sum_148, sum_149, permute_840, permute_836, permute_832, permute_819, permute_815, view_1190, permute_811, view_1187, sum_141, sum_142, permute_807, permute_803, permute_799, permute_786, permute_782, view_1164, permute_778, view_1161, sum_134, sum_135, permute_774, permute_770, permute_766, permute_753, permute_749, view_1138, permute_745, view_1135, sum_127, sum_128, permute_741, permute_737, permute_733, permute_720, permute_716, view_1112, permute_712, view_1109, sum_120, sum_121, permute_708, permute_704, permute_700, permute_687, permute_683, view_1086, permute_679, view_1083, sum_113, sum_114, permute_675, permute_671, permute_667, permute_654, permute_650, view_1060, permute_646, view_1057, sum_106, sum_107, permute_642, permute_638, permute_634, permute_621, permute_617, view_1034, permute_613, view_1031, sum_99, sum_100, permute_609, permute_605, permute_601, permute_588, permute_584, view_1008, permute_580, view_1005, sum_92, sum_93, permute_576, permute_572, permute_568, permute_555, permute_551, view_982, permute_547, view_979, sum_85, sum_86, permute_543, permute_539, permute_535, permute_522, permute_518, view_956, permute_514, view_953, sum_78, sum_79, permute_510, permute_506, permute_502, permute_489, permute_485, view_930, permute_481, view_927, sum_71, sum_72, permute_477, permute_473, permute_469, permute_456, permute_452, view_904, permute_448, view_901, sum_64, sum_65, permute_444, permute_440, permute_436, permute_423, permute_419, view_878, permute_415, view_875, sum_57, sum_58, permute_411, permute_407, permute_403, permute_390, permute_386, view_852, permute_382, view_849, sum_50, sum_51, permute_378, permute_374, permute_370, permute_357, permute_353, view_826, permute_349, view_823, sum_43, sum_44, permute_345, permute_341, permute_337, permute_324, permute_320, view_800, permute_316, view_797, sum_36, sum_37, permute_312, view_793, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    