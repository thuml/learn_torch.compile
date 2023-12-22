from __future__ import annotations



def forward(self, primals_2: "f32[4096]", primals_12: "f32[4096]", primals_22: "f32[4096]", primals_32: "f32[4096]", primals_42: "f32[4096]", primals_52: "f32[4096]", primals_62: "f32[4096]", primals_72: "f32[4096]", primals_82: "f32[4096]", primals_92: "f32[4096]", primals_102: "f32[4096]", primals_112: "f32[4096]", primals_122: "f32[4096]", primals_132: "f32[4096]", primals_142: "f32[4096]", primals_152: "f32[4096]", primals_162: "f32[4096]", primals_172: "f32[4096]", primals_182: "f32[4096]", primals_192: "f32[4096]", primals_202: "f32[4096]", primals_212: "f32[4096]", primals_222: "f32[4096]", primals_232: "f32[4096]", primals_242: "f32[4096]", primals_252: "f32[4096]", primals_262: "f32[4096]", primals_272: "f32[4096]", primals_282: "f32[4096]", primals_288: "f32[]", primals_291: "f32[]", primals_294: "f32[]", primals_297: "f32[]", primals_300: "f32[]", primals_303: "f32[]", primals_306: "f32[]", primals_309: "f32[]", primals_312: "f32[]", primals_315: "f32[]", primals_318: "f32[]", primals_321: "f32[]", primals_324: "f32[]", primals_327: "f32[]", primals_330: "f32[]", primals_333: "f32[]", primals_336: "f32[]", primals_339: "f32[]", primals_342: "f32[]", primals_345: "f32[]", primals_348: "f32[]", primals_351: "f32[]", primals_354: "f32[]", primals_357: "f32[]", primals_360: "f32[]", primals_363: "f32[]", primals_366: "f32[]", primals_369: "f32[]", view: "i64[1, 128]", embedding: "f32[1, 128, 4096]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", view_2: "f32[128, 4096]", unsqueeze_3: "f32[1, 128, 1, 32, 1]", unsqueeze_5: "f32[1, 128, 1, 32, 1]", slice_48: "b8[1, 1, 128, 128]", view_24: "f32[128, 4096]", addmm: "f32[128, 16384]", tanh: "f32[1, 128, 16384]", view_28: "f32[128, 16384]", mul_10: "f32[1, 128, 4096]", view_30: "f32[128, 4096]", unsqueeze_16: "f32[1, 128, 1, 32, 1]", unsqueeze_18: "f32[1, 128, 1, 32, 1]", slice_96: "b8[1, 1, 128, 128]", view_52: "f32[128, 4096]", addmm_2: "f32[128, 16384]", tanh_1: "f32[1, 128, 16384]", view_56: "f32[128, 16384]", mul_20: "f32[1, 128, 4096]", view_58: "f32[128, 4096]", unsqueeze_29: "f32[1, 128, 1, 32, 1]", unsqueeze_31: "f32[1, 128, 1, 32, 1]", slice_144: "b8[1, 1, 128, 128]", view_80: "f32[128, 4096]", addmm_4: "f32[128, 16384]", tanh_2: "f32[1, 128, 16384]", view_84: "f32[128, 16384]", mul_30: "f32[1, 128, 4096]", view_86: "f32[128, 4096]", unsqueeze_42: "f32[1, 128, 1, 32, 1]", unsqueeze_44: "f32[1, 128, 1, 32, 1]", slice_192: "b8[1, 1, 128, 128]", view_108: "f32[128, 4096]", addmm_6: "f32[128, 16384]", tanh_3: "f32[1, 128, 16384]", view_112: "f32[128, 16384]", mul_40: "f32[1, 128, 4096]", view_114: "f32[128, 4096]", unsqueeze_55: "f32[1, 128, 1, 32, 1]", unsqueeze_57: "f32[1, 128, 1, 32, 1]", slice_240: "b8[1, 1, 128, 128]", view_136: "f32[128, 4096]", addmm_8: "f32[128, 16384]", tanh_4: "f32[1, 128, 16384]", view_140: "f32[128, 16384]", mul_50: "f32[1, 128, 4096]", view_142: "f32[128, 4096]", unsqueeze_68: "f32[1, 128, 1, 32, 1]", unsqueeze_70: "f32[1, 128, 1, 32, 1]", slice_288: "b8[1, 1, 128, 128]", view_164: "f32[128, 4096]", addmm_10: "f32[128, 16384]", tanh_5: "f32[1, 128, 16384]", view_168: "f32[128, 16384]", mul_60: "f32[1, 128, 4096]", view_170: "f32[128, 4096]", unsqueeze_81: "f32[1, 128, 1, 32, 1]", unsqueeze_83: "f32[1, 128, 1, 32, 1]", slice_336: "b8[1, 1, 128, 128]", view_192: "f32[128, 4096]", addmm_12: "f32[128, 16384]", tanh_6: "f32[1, 128, 16384]", view_196: "f32[128, 16384]", mul_70: "f32[1, 128, 4096]", view_198: "f32[128, 4096]", unsqueeze_94: "f32[1, 128, 1, 32, 1]", unsqueeze_96: "f32[1, 128, 1, 32, 1]", slice_384: "b8[1, 1, 128, 128]", view_220: "f32[128, 4096]", addmm_14: "f32[128, 16384]", tanh_7: "f32[1, 128, 16384]", view_224: "f32[128, 16384]", mul_80: "f32[1, 128, 4096]", view_226: "f32[128, 4096]", unsqueeze_107: "f32[1, 128, 1, 32, 1]", unsqueeze_109: "f32[1, 128, 1, 32, 1]", slice_432: "b8[1, 1, 128, 128]", view_248: "f32[128, 4096]", addmm_16: "f32[128, 16384]", tanh_8: "f32[1, 128, 16384]", view_252: "f32[128, 16384]", mul_90: "f32[1, 128, 4096]", view_254: "f32[128, 4096]", unsqueeze_120: "f32[1, 128, 1, 32, 1]", unsqueeze_122: "f32[1, 128, 1, 32, 1]", slice_480: "b8[1, 1, 128, 128]", view_276: "f32[128, 4096]", addmm_18: "f32[128, 16384]", tanh_9: "f32[1, 128, 16384]", view_280: "f32[128, 16384]", mul_100: "f32[1, 128, 4096]", view_282: "f32[128, 4096]", unsqueeze_133: "f32[1, 128, 1, 32, 1]", unsqueeze_135: "f32[1, 128, 1, 32, 1]", slice_528: "b8[1, 1, 128, 128]", view_304: "f32[128, 4096]", addmm_20: "f32[128, 16384]", tanh_10: "f32[1, 128, 16384]", view_308: "f32[128, 16384]", mul_110: "f32[1, 128, 4096]", view_310: "f32[128, 4096]", unsqueeze_146: "f32[1, 128, 1, 32, 1]", unsqueeze_148: "f32[1, 128, 1, 32, 1]", slice_576: "b8[1, 1, 128, 128]", view_332: "f32[128, 4096]", addmm_22: "f32[128, 16384]", tanh_11: "f32[1, 128, 16384]", view_336: "f32[128, 16384]", mul_120: "f32[1, 128, 4096]", view_338: "f32[128, 4096]", unsqueeze_159: "f32[1, 128, 1, 32, 1]", unsqueeze_161: "f32[1, 128, 1, 32, 1]", slice_624: "b8[1, 1, 128, 128]", view_360: "f32[128, 4096]", addmm_24: "f32[128, 16384]", tanh_12: "f32[1, 128, 16384]", view_364: "f32[128, 16384]", mul_130: "f32[1, 128, 4096]", view_366: "f32[128, 4096]", unsqueeze_172: "f32[1, 128, 1, 32, 1]", unsqueeze_174: "f32[1, 128, 1, 32, 1]", slice_672: "b8[1, 1, 128, 128]", view_388: "f32[128, 4096]", addmm_26: "f32[128, 16384]", tanh_13: "f32[1, 128, 16384]", view_392: "f32[128, 16384]", mul_140: "f32[1, 128, 4096]", view_394: "f32[128, 4096]", unsqueeze_185: "f32[1, 128, 1, 32, 1]", unsqueeze_187: "f32[1, 128, 1, 32, 1]", slice_720: "b8[1, 1, 128, 128]", view_416: "f32[128, 4096]", addmm_28: "f32[128, 16384]", tanh_14: "f32[1, 128, 16384]", view_420: "f32[128, 16384]", mul_150: "f32[1, 128, 4096]", view_422: "f32[128, 4096]", unsqueeze_198: "f32[1, 128, 1, 32, 1]", unsqueeze_200: "f32[1, 128, 1, 32, 1]", slice_768: "b8[1, 1, 128, 128]", view_444: "f32[128, 4096]", addmm_30: "f32[128, 16384]", tanh_15: "f32[1, 128, 16384]", view_448: "f32[128, 16384]", mul_160: "f32[1, 128, 4096]", view_450: "f32[128, 4096]", unsqueeze_211: "f32[1, 128, 1, 32, 1]", unsqueeze_213: "f32[1, 128, 1, 32, 1]", slice_816: "b8[1, 1, 128, 128]", view_472: "f32[128, 4096]", addmm_32: "f32[128, 16384]", tanh_16: "f32[1, 128, 16384]", view_476: "f32[128, 16384]", mul_170: "f32[1, 128, 4096]", view_478: "f32[128, 4096]", unsqueeze_224: "f32[1, 128, 1, 32, 1]", unsqueeze_226: "f32[1, 128, 1, 32, 1]", slice_864: "b8[1, 1, 128, 128]", view_500: "f32[128, 4096]", addmm_34: "f32[128, 16384]", tanh_17: "f32[1, 128, 16384]", view_504: "f32[128, 16384]", mul_180: "f32[1, 128, 4096]", view_506: "f32[128, 4096]", unsqueeze_237: "f32[1, 128, 1, 32, 1]", unsqueeze_239: "f32[1, 128, 1, 32, 1]", slice_912: "b8[1, 1, 128, 128]", view_528: "f32[128, 4096]", addmm_36: "f32[128, 16384]", tanh_18: "f32[1, 128, 16384]", view_532: "f32[128, 16384]", mul_190: "f32[1, 128, 4096]", view_534: "f32[128, 4096]", unsqueeze_250: "f32[1, 128, 1, 32, 1]", unsqueeze_252: "f32[1, 128, 1, 32, 1]", slice_960: "b8[1, 1, 128, 128]", view_556: "f32[128, 4096]", addmm_38: "f32[128, 16384]", tanh_19: "f32[1, 128, 16384]", view_560: "f32[128, 16384]", mul_200: "f32[1, 128, 4096]", view_562: "f32[128, 4096]", unsqueeze_263: "f32[1, 128, 1, 32, 1]", unsqueeze_265: "f32[1, 128, 1, 32, 1]", slice_1008: "b8[1, 1, 128, 128]", view_584: "f32[128, 4096]", addmm_40: "f32[128, 16384]", tanh_20: "f32[1, 128, 16384]", view_588: "f32[128, 16384]", mul_210: "f32[1, 128, 4096]", view_590: "f32[128, 4096]", unsqueeze_276: "f32[1, 128, 1, 32, 1]", unsqueeze_278: "f32[1, 128, 1, 32, 1]", slice_1056: "b8[1, 1, 128, 128]", view_612: "f32[128, 4096]", addmm_42: "f32[128, 16384]", tanh_21: "f32[1, 128, 16384]", view_616: "f32[128, 16384]", mul_220: "f32[1, 128, 4096]", view_618: "f32[128, 4096]", unsqueeze_289: "f32[1, 128, 1, 32, 1]", unsqueeze_291: "f32[1, 128, 1, 32, 1]", slice_1104: "b8[1, 1, 128, 128]", view_640: "f32[128, 4096]", addmm_44: "f32[128, 16384]", tanh_22: "f32[1, 128, 16384]", view_644: "f32[128, 16384]", mul_230: "f32[1, 128, 4096]", view_646: "f32[128, 4096]", unsqueeze_302: "f32[1, 128, 1, 32, 1]", unsqueeze_304: "f32[1, 128, 1, 32, 1]", slice_1152: "b8[1, 1, 128, 128]", view_668: "f32[128, 4096]", addmm_46: "f32[128, 16384]", tanh_23: "f32[1, 128, 16384]", view_672: "f32[128, 16384]", mul_240: "f32[1, 128, 4096]", view_674: "f32[128, 4096]", unsqueeze_315: "f32[1, 128, 1, 32, 1]", unsqueeze_317: "f32[1, 128, 1, 32, 1]", slice_1200: "b8[1, 1, 128, 128]", view_696: "f32[128, 4096]", addmm_48: "f32[128, 16384]", tanh_24: "f32[1, 128, 16384]", view_700: "f32[128, 16384]", mul_250: "f32[1, 128, 4096]", view_702: "f32[128, 4096]", unsqueeze_328: "f32[1, 128, 1, 32, 1]", unsqueeze_330: "f32[1, 128, 1, 32, 1]", slice_1248: "b8[1, 1, 128, 128]", view_724: "f32[128, 4096]", addmm_50: "f32[128, 16384]", tanh_25: "f32[1, 128, 16384]", view_728: "f32[128, 16384]", mul_260: "f32[1, 128, 4096]", view_730: "f32[128, 4096]", unsqueeze_341: "f32[1, 128, 1, 32, 1]", unsqueeze_343: "f32[1, 128, 1, 32, 1]", slice_1296: "b8[1, 1, 128, 128]", view_752: "f32[128, 4096]", addmm_52: "f32[128, 16384]", tanh_26: "f32[1, 128, 16384]", view_756: "f32[128, 16384]", mul_270: "f32[1, 128, 4096]", view_758: "f32[128, 4096]", unsqueeze_354: "f32[1, 128, 1, 32, 1]", unsqueeze_356: "f32[1, 128, 1, 32, 1]", slice_1344: "b8[1, 1, 128, 128]", view_780: "f32[128, 4096]", addmm_54: "f32[128, 16384]", tanh_27: "f32[1, 128, 16384]", view_784: "f32[128, 16384]", mul_280: "f32[1, 128, 4096]", view_787: "f32[128, 4096]", sub_58: "f32[1, 128]", ne: "b8[1]", sub_60: "f32[1, 128]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_32: "i64[1, 1]", ne_8: "b8[1, 1]", where_34: "i64[1, 1]", permute_309: "f32[2, 4096]", div_62: "f32[1, 128, 1]", permute_313: "f32[4096, 16384]", permute_317: "f32[16384, 4096]", permute_323: "f32[4096, 4096]", permute_326: "f32[16, 128, 128]", permute_327: "f32[16, 256, 128]", alias_61: "f32[1, 16, 128, 128]", permute_328: "f32[16, 256, 128]", permute_329: "f32[16, 128, 256]", permute_336: "f32[4096, 4096]", permute_340: "f32[4096, 4096]", permute_344: "f32[4096, 4096]", div_64: "f32[1, 128, 1]", permute_346: "f32[4096, 16384]", permute_350: "f32[16384, 4096]", permute_356: "f32[4096, 4096]", permute_359: "f32[16, 128, 128]", permute_360: "f32[16, 256, 128]", alias_63: "f32[1, 16, 128, 128]", permute_361: "f32[16, 256, 128]", permute_362: "f32[16, 128, 256]", permute_369: "f32[4096, 4096]", permute_373: "f32[4096, 4096]", permute_377: "f32[4096, 4096]", div_66: "f32[1, 128, 1]", permute_379: "f32[4096, 16384]", permute_383: "f32[16384, 4096]", permute_389: "f32[4096, 4096]", permute_392: "f32[16, 128, 128]", permute_393: "f32[16, 256, 128]", alias_65: "f32[1, 16, 128, 128]", permute_394: "f32[16, 256, 128]", permute_395: "f32[16, 128, 256]", permute_402: "f32[4096, 4096]", permute_406: "f32[4096, 4096]", permute_410: "f32[4096, 4096]", div_68: "f32[1, 128, 1]", permute_412: "f32[4096, 16384]", permute_416: "f32[16384, 4096]", permute_422: "f32[4096, 4096]", permute_425: "f32[16, 128, 128]", permute_426: "f32[16, 256, 128]", alias_67: "f32[1, 16, 128, 128]", permute_427: "f32[16, 256, 128]", permute_428: "f32[16, 128, 256]", permute_435: "f32[4096, 4096]", permute_439: "f32[4096, 4096]", permute_443: "f32[4096, 4096]", div_70: "f32[1, 128, 1]", permute_445: "f32[4096, 16384]", permute_449: "f32[16384, 4096]", permute_455: "f32[4096, 4096]", permute_458: "f32[16, 128, 128]", permute_459: "f32[16, 256, 128]", alias_69: "f32[1, 16, 128, 128]", permute_460: "f32[16, 256, 128]", permute_461: "f32[16, 128, 256]", permute_468: "f32[4096, 4096]", permute_472: "f32[4096, 4096]", permute_476: "f32[4096, 4096]", div_72: "f32[1, 128, 1]", permute_478: "f32[4096, 16384]", permute_482: "f32[16384, 4096]", permute_488: "f32[4096, 4096]", permute_491: "f32[16, 128, 128]", permute_492: "f32[16, 256, 128]", alias_71: "f32[1, 16, 128, 128]", permute_493: "f32[16, 256, 128]", permute_494: "f32[16, 128, 256]", permute_501: "f32[4096, 4096]", permute_505: "f32[4096, 4096]", permute_509: "f32[4096, 4096]", div_74: "f32[1, 128, 1]", permute_511: "f32[4096, 16384]", permute_515: "f32[16384, 4096]", permute_521: "f32[4096, 4096]", permute_524: "f32[16, 128, 128]", permute_525: "f32[16, 256, 128]", alias_73: "f32[1, 16, 128, 128]", permute_526: "f32[16, 256, 128]", permute_527: "f32[16, 128, 256]", permute_534: "f32[4096, 4096]", permute_538: "f32[4096, 4096]", permute_542: "f32[4096, 4096]", div_76: "f32[1, 128, 1]", permute_544: "f32[4096, 16384]", permute_548: "f32[16384, 4096]", permute_554: "f32[4096, 4096]", permute_557: "f32[16, 128, 128]", permute_558: "f32[16, 256, 128]", alias_75: "f32[1, 16, 128, 128]", permute_559: "f32[16, 256, 128]", permute_560: "f32[16, 128, 256]", permute_567: "f32[4096, 4096]", permute_571: "f32[4096, 4096]", permute_575: "f32[4096, 4096]", div_78: "f32[1, 128, 1]", permute_577: "f32[4096, 16384]", permute_581: "f32[16384, 4096]", permute_587: "f32[4096, 4096]", permute_590: "f32[16, 128, 128]", permute_591: "f32[16, 256, 128]", alias_77: "f32[1, 16, 128, 128]", permute_592: "f32[16, 256, 128]", permute_593: "f32[16, 128, 256]", permute_600: "f32[4096, 4096]", permute_604: "f32[4096, 4096]", permute_608: "f32[4096, 4096]", div_80: "f32[1, 128, 1]", permute_610: "f32[4096, 16384]", permute_614: "f32[16384, 4096]", permute_620: "f32[4096, 4096]", permute_623: "f32[16, 128, 128]", permute_624: "f32[16, 256, 128]", alias_79: "f32[1, 16, 128, 128]", permute_625: "f32[16, 256, 128]", permute_626: "f32[16, 128, 256]", permute_633: "f32[4096, 4096]", permute_637: "f32[4096, 4096]", permute_641: "f32[4096, 4096]", div_82: "f32[1, 128, 1]", permute_643: "f32[4096, 16384]", permute_647: "f32[16384, 4096]", permute_653: "f32[4096, 4096]", permute_656: "f32[16, 128, 128]", permute_657: "f32[16, 256, 128]", alias_81: "f32[1, 16, 128, 128]", permute_658: "f32[16, 256, 128]", permute_659: "f32[16, 128, 256]", permute_666: "f32[4096, 4096]", permute_670: "f32[4096, 4096]", permute_674: "f32[4096, 4096]", div_84: "f32[1, 128, 1]", permute_676: "f32[4096, 16384]", permute_680: "f32[16384, 4096]", permute_686: "f32[4096, 4096]", permute_689: "f32[16, 128, 128]", permute_690: "f32[16, 256, 128]", alias_83: "f32[1, 16, 128, 128]", permute_691: "f32[16, 256, 128]", permute_692: "f32[16, 128, 256]", permute_699: "f32[4096, 4096]", permute_703: "f32[4096, 4096]", permute_707: "f32[4096, 4096]", div_86: "f32[1, 128, 1]", permute_709: "f32[4096, 16384]", permute_713: "f32[16384, 4096]", permute_719: "f32[4096, 4096]", permute_722: "f32[16, 128, 128]", permute_723: "f32[16, 256, 128]", alias_85: "f32[1, 16, 128, 128]", permute_724: "f32[16, 256, 128]", permute_725: "f32[16, 128, 256]", permute_732: "f32[4096, 4096]", permute_736: "f32[4096, 4096]", permute_740: "f32[4096, 4096]", div_88: "f32[1, 128, 1]", permute_742: "f32[4096, 16384]", permute_746: "f32[16384, 4096]", permute_752: "f32[4096, 4096]", permute_755: "f32[16, 128, 128]", permute_756: "f32[16, 256, 128]", alias_87: "f32[1, 16, 128, 128]", permute_757: "f32[16, 256, 128]", permute_758: "f32[16, 128, 256]", permute_765: "f32[4096, 4096]", permute_769: "f32[4096, 4096]", permute_773: "f32[4096, 4096]", div_90: "f32[1, 128, 1]", permute_775: "f32[4096, 16384]", permute_779: "f32[16384, 4096]", permute_785: "f32[4096, 4096]", permute_788: "f32[16, 128, 128]", permute_789: "f32[16, 256, 128]", alias_89: "f32[1, 16, 128, 128]", permute_790: "f32[16, 256, 128]", permute_791: "f32[16, 128, 256]", permute_798: "f32[4096, 4096]", permute_802: "f32[4096, 4096]", permute_806: "f32[4096, 4096]", div_92: "f32[1, 128, 1]", permute_808: "f32[4096, 16384]", permute_812: "f32[16384, 4096]", permute_818: "f32[4096, 4096]", permute_821: "f32[16, 128, 128]", permute_822: "f32[16, 256, 128]", alias_91: "f32[1, 16, 128, 128]", permute_823: "f32[16, 256, 128]", permute_824: "f32[16, 128, 256]", permute_831: "f32[4096, 4096]", permute_835: "f32[4096, 4096]", permute_839: "f32[4096, 4096]", div_94: "f32[1, 128, 1]", permute_841: "f32[4096, 16384]", permute_845: "f32[16384, 4096]", permute_851: "f32[4096, 4096]", permute_854: "f32[16, 128, 128]", permute_855: "f32[16, 256, 128]", alias_93: "f32[1, 16, 128, 128]", permute_856: "f32[16, 256, 128]", permute_857: "f32[16, 128, 256]", permute_864: "f32[4096, 4096]", permute_868: "f32[4096, 4096]", permute_872: "f32[4096, 4096]", div_96: "f32[1, 128, 1]", permute_874: "f32[4096, 16384]", permute_878: "f32[16384, 4096]", permute_884: "f32[4096, 4096]", permute_887: "f32[16, 128, 128]", permute_888: "f32[16, 256, 128]", alias_95: "f32[1, 16, 128, 128]", permute_889: "f32[16, 256, 128]", permute_890: "f32[16, 128, 256]", permute_897: "f32[4096, 4096]", permute_901: "f32[4096, 4096]", permute_905: "f32[4096, 4096]", div_98: "f32[1, 128, 1]", permute_907: "f32[4096, 16384]", permute_911: "f32[16384, 4096]", permute_917: "f32[4096, 4096]", permute_920: "f32[16, 128, 128]", permute_921: "f32[16, 256, 128]", alias_97: "f32[1, 16, 128, 128]", permute_922: "f32[16, 256, 128]", permute_923: "f32[16, 128, 256]", permute_930: "f32[4096, 4096]", permute_934: "f32[4096, 4096]", permute_938: "f32[4096, 4096]", div_100: "f32[1, 128, 1]", permute_940: "f32[4096, 16384]", permute_944: "f32[16384, 4096]", permute_950: "f32[4096, 4096]", permute_953: "f32[16, 128, 128]", permute_954: "f32[16, 256, 128]", alias_99: "f32[1, 16, 128, 128]", permute_955: "f32[16, 256, 128]", permute_956: "f32[16, 128, 256]", permute_963: "f32[4096, 4096]", permute_967: "f32[4096, 4096]", permute_971: "f32[4096, 4096]", div_102: "f32[1, 128, 1]", permute_973: "f32[4096, 16384]", permute_977: "f32[16384, 4096]", permute_983: "f32[4096, 4096]", permute_986: "f32[16, 128, 128]", permute_987: "f32[16, 256, 128]", alias_101: "f32[1, 16, 128, 128]", permute_988: "f32[16, 256, 128]", permute_989: "f32[16, 128, 256]", permute_996: "f32[4096, 4096]", permute_1000: "f32[4096, 4096]", permute_1004: "f32[4096, 4096]", div_104: "f32[1, 128, 1]", permute_1006: "f32[4096, 16384]", permute_1010: "f32[16384, 4096]", permute_1016: "f32[4096, 4096]", permute_1019: "f32[16, 128, 128]", permute_1020: "f32[16, 256, 128]", alias_103: "f32[1, 16, 128, 128]", permute_1021: "f32[16, 256, 128]", permute_1022: "f32[16, 128, 256]", permute_1029: "f32[4096, 4096]", permute_1033: "f32[4096, 4096]", permute_1037: "f32[4096, 4096]", div_106: "f32[1, 128, 1]", permute_1039: "f32[4096, 16384]", permute_1043: "f32[16384, 4096]", permute_1049: "f32[4096, 4096]", permute_1052: "f32[16, 128, 128]", permute_1053: "f32[16, 256, 128]", alias_105: "f32[1, 16, 128, 128]", permute_1054: "f32[16, 256, 128]", permute_1055: "f32[16, 128, 256]", permute_1062: "f32[4096, 4096]", permute_1066: "f32[4096, 4096]", permute_1070: "f32[4096, 4096]", div_108: "f32[1, 128, 1]", permute_1072: "f32[4096, 16384]", permute_1076: "f32[16384, 4096]", permute_1082: "f32[4096, 4096]", permute_1085: "f32[16, 128, 128]", permute_1086: "f32[16, 256, 128]", alias_107: "f32[1, 16, 128, 128]", permute_1087: "f32[16, 256, 128]", permute_1088: "f32[16, 128, 256]", permute_1095: "f32[4096, 4096]", permute_1099: "f32[4096, 4096]", permute_1103: "f32[4096, 4096]", div_110: "f32[1, 128, 1]", permute_1105: "f32[4096, 16384]", permute_1109: "f32[16384, 4096]", permute_1115: "f32[4096, 4096]", permute_1118: "f32[16, 128, 128]", permute_1119: "f32[16, 256, 128]", alias_109: "f32[1, 16, 128, 128]", permute_1120: "f32[16, 256, 128]", permute_1121: "f32[16, 128, 256]", permute_1128: "f32[4096, 4096]", permute_1132: "f32[4096, 4096]", permute_1136: "f32[4096, 4096]", div_112: "f32[1, 128, 1]", permute_1138: "f32[4096, 16384]", permute_1142: "f32[16384, 4096]", permute_1148: "f32[4096, 4096]", permute_1151: "f32[16, 128, 128]", permute_1152: "f32[16, 256, 128]", alias_111: "f32[1, 16, 128, 128]", permute_1153: "f32[16, 256, 128]", permute_1154: "f32[16, 128, 256]", permute_1161: "f32[4096, 4096]", permute_1165: "f32[4096, 4096]", permute_1169: "f32[4096, 4096]", div_114: "f32[1, 128, 1]", permute_1171: "f32[4096, 16384]", permute_1175: "f32[16384, 4096]", permute_1181: "f32[4096, 4096]", permute_1184: "f32[16, 128, 128]", permute_1185: "f32[16, 256, 128]", alias_113: "f32[1, 16, 128, 128]", permute_1186: "f32[16, 256, 128]", permute_1187: "f32[16, 128, 256]", permute_1194: "f32[4096, 4096]", permute_1198: "f32[4096, 4096]", permute_1202: "f32[4096, 4096]", div_116: "f32[1, 128, 1]", permute_1204: "f32[4096, 16384]", permute_1208: "f32[16384, 4096]", permute_1214: "f32[4096, 4096]", permute_1217: "f32[16, 128, 128]", permute_1218: "f32[16, 256, 128]", alias_115: "f32[1, 16, 128, 128]", permute_1219: "f32[16, 256, 128]", permute_1220: "f32[16, 128, 256]", permute_1227: "f32[4096, 4096]", permute_1231: "f32[4096, 4096]", permute_1235: "f32[4096, 4096]", tangents_1: "f32[]", tangents_2: "f32[1, 128]", tangents_3: "f32[1, 128]"):
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sum_30: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    sum_33: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_33, torch.float32);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1142, code: total_loss = (start_loss + end_loss) / 2
    div_59: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    div_60: "f32[]" = torch.ops.aten.div.Tensor(div_59, convert_element_type_1);  convert_element_type_1 = None
    full_default_33: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 128]" = torch.ops.aten.scatter.value(full_default_33, 1, where_32, -1.0);  where_32 = None
    where_33: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_60, full_default_29);  ne_6 = div_60 = None
    mul_282: "f32[1, 128]" = torch.ops.aten.mul.Tensor(scatter, where_33);  scatter = where_33 = None
    exp_30: "f32[1, 128]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_35: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [1], True)
    mul_283: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_30, sum_35);  exp_30 = sum_35 = None
    sub_61: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1141, code: end_loss = loss_fct(end_logits, end_positions)
    add_227: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_3, sub_61);  tangents_3 = sub_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    div_61: "f32[]" = torch.ops.aten.div.Tensor(div_59, convert_element_type);  div_59 = convert_element_type = None
    scatter_1: "f32[1, 128]" = torch.ops.aten.scatter.value(full_default_33, 1, where_34, -1.0);  full_default_33 = where_34 = None
    where_35: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_61, full_default_29);  ne_8 = div_61 = None
    mul_284: "f32[1, 128]" = torch.ops.aten.mul.Tensor(scatter_1, where_35);  scatter_1 = where_35 = None
    exp_31: "f32[1, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_36: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [1], True)
    mul_285: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_31, sum_36);  exp_31 = sum_36 = None
    sub_62: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1140, code: start_loss = loss_fct(start_logits, start_positions)
    add_228: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_2, sub_62);  tangents_2 = sub_62 = None
    
    # No stacktrace found for following nodes
    unsqueeze_369: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_227, 2);  add_227 = None
    unsqueeze_370: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_228, 2);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1123, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat_112: "f32[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_370, unsqueeze_369], 2);  unsqueeze_370 = unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:1122, code: logits = self.qa_outputs(sequence_output)
    view_789: "f32[128, 2]" = torch.ops.aten.reshape.default(cat_112, [128, 2]);  cat_112 = None
    mm_112: "f32[128, 4096]" = torch.ops.aten.mm.default(view_789, permute_309);  permute_309 = None
    permute_310: "f32[2, 128]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_113: "f32[2, 4096]" = torch.ops.aten.mm.default(permute_310, view_787);  permute_310 = view_787 = None
    permute_311: "f32[4096, 2]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_37: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[2]" = torch.ops.aten.reshape.default(sum_37, [2]);  sum_37 = None
    permute_312: "f32[2, 4096]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_791: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_112, [1, 128, 4096]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    mul_287: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_791, primals_282);  primals_282 = None
    mul_288: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_287, 4096)
    sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
    mul_289: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_287, mul_280);  mul_287 = None
    sum_39: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
    mul_290: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, sum_39);  sum_39 = None
    sub_64: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_288, sum_38);  mul_288 = sum_38 = None
    sub_65: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_64, mul_290);  sub_64 = mul_290 = None
    mul_291: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_62, sub_65);  div_62 = sub_65 = None
    mul_292: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_791, mul_280);  mul_280 = None
    sum_40: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
    sum_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_791, [0, 1]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_793: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_291, [128, 4096])
    mm_114: "f32[128, 16384]" = torch.ops.aten.mm.default(view_793, permute_313);  permute_313 = None
    permute_314: "f32[4096, 128]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_115: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_314, view_784);  view_784 = None
    permute_315: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_42: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True)
    view_794: "f32[4096]" = torch.ops.aten.reshape.default(sum_42, [4096]);  sum_42 = None
    permute_316: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    view_795: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_114, [1, 128, 16384]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_293: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_795, mul_276);  mul_276 = None
    mul_294: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_795, add_221);  view_795 = add_221 = None
    mul_295: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
    sub_66: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_295);  mul_295 = None
    mul_296: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_293, sub_66);  mul_293 = sub_66 = None
    mul_297: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_296, 0.7978845608028654);  mul_296 = None
    mul_298: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_297, 0.044715)
    pow_29: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_783, 2.0);  view_783 = None
    mul_299: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
    mul_300: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_229: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_297, mul_300);  mul_297 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_301: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_294, 0.5);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_230: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_229, mul_301);  add_229 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_796: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_230, [128, 16384]);  add_230 = None
    mm_116: "f32[128, 4096]" = torch.ops.aten.mm.default(view_796, permute_317);  permute_317 = None
    permute_318: "f32[16384, 128]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_117: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_318, view_758);  permute_318 = None
    permute_319: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_43: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[16384]" = torch.ops.aten.reshape.default(sum_43, [16384]);  sum_43 = None
    permute_320: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_798: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_116, [1, 128, 4096]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_118: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_314, view_780);  permute_314 = view_780 = None
    permute_322: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    mm_119: "f32[128, 4096]" = torch.ops.aten.mm.default(view_793, permute_323);  view_793 = permute_323 = None
    view_800: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 4096]);  mm_119 = None
    permute_324: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_801: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_800, [1, 128, 16, 256]);  view_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_325: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_802: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_325, [16, 128, 256]);  permute_325 = None
    bmm_56: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_326, view_802);  permute_326 = None
    bmm_57: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_802, permute_327);  view_802 = permute_327 = None
    view_803: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 256]);  bmm_56 = None
    view_804: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_302: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_804, alias_61);  view_804 = None
    sum_44: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [-1], True)
    mul_303: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_61, sum_44);  alias_61 = sum_44 = None
    sub_67: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_63: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_67, primals_369);  sub_67 = primals_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_36: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1344, div_63, full_default_29);  slice_1344 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_805: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_36, [16, 128, 128]);  where_36 = None
    bmm_58: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_328, view_805);  permute_328 = None
    bmm_59: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_805, permute_329);  view_805 = permute_329 = None
    view_806: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_58, [1, 16, 256, 128]);  bmm_58 = None
    view_807: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_59, [1, 16, 128, 256]);  bmm_59 = None
    permute_330: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_806, [0, 1, 3, 2]);  view_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_331: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_807, [0, 2, 1, 3]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_332: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_330, [0, 2, 1, 3]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1345: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_331, 3, 0, 64)
    slice_1346: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_331, 3, 64, 256);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1347: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64)
    slice_1348: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 256);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_304: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1345, view_767)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_808: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_304, [1, 128, 16, 32, 2]);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_808, 4, 0)
    select_1: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_808, 4, 1);  view_808 = None
    neg_58: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    full_default_39: "f32[1, 128, 16, 64]" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_4: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_231: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_4);  slice_scatter = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_305: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1345, view_768);  slice_1345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_232: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_231, mul_305);  add_231 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_306: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1347, view_767);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_809: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_306, [1, 128, 16, 32, 2]);  mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_2: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_809, 4, 0)
    select_3: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_809, 4, 1);  view_809 = None
    neg_59: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_2);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_8: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_12: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_233: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_12);  slice_scatter_8 = slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_307: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1347, view_768);  slice_1347 = view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_234: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_233, mul_307);  add_233 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    full_default_55: "f32[1, 128, 16, 256]" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1346, 3, 64, 9223372036854775807);  slice_1346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_20: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_232, 3, 0, 64);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_235: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_20);  slice_scatter_16 = slice_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_24: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1348, 3, 64, 9223372036854775807);  slice_1348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_28: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_234, 3, 0, 64);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_236: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_28);  slice_scatter_24 = slice_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_333: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_803, [0, 2, 1, 3]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_227: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    view_810: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_227, [1, 128, 4096]);  clone_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_811: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_236, [1, 128, 4096]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_812: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_235, [1, 128, 4096]);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_813: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_810, [128, 4096]);  view_810 = None
    permute_334: "f32[4096, 128]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_120: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_334, view_758);  permute_334 = None
    permute_335: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    mm_121: "f32[128, 4096]" = torch.ops.aten.mm.default(view_813, permute_336);  view_813 = permute_336 = None
    view_814: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 4096]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_237: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_798, view_814);  view_798 = view_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_337: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_815: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_811, [128, 4096]);  view_811 = None
    permute_338: "f32[4096, 128]" = torch.ops.aten.permute.default(view_815, [1, 0])
    mm_122: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_338, view_758);  permute_338 = None
    permute_339: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    mm_123: "f32[128, 4096]" = torch.ops.aten.mm.default(view_815, permute_340);  view_815 = permute_340 = None
    view_816: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_238: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_237, view_816);  add_237 = view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_341: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_817: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_812, [128, 4096]);  view_812 = None
    permute_342: "f32[4096, 128]" = torch.ops.aten.permute.default(view_817, [1, 0])
    mm_124: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_342, view_758);  permute_342 = view_758 = None
    permute_343: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    mm_125: "f32[128, 4096]" = torch.ops.aten.mm.default(view_817, permute_344);  view_817 = permute_344 = None
    view_818: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 4096]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_239: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_238, view_818);  add_238 = view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_345: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_309: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_239, primals_272);  primals_272 = None
    mul_310: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_309, 4096)
    sum_45: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_309, mul_270);  mul_309 = None
    sum_46: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, sum_46);  sum_46 = None
    sub_69: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_310, sum_45);  mul_310 = sum_45 = None
    sub_70: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_69, mul_312);  sub_69 = mul_312 = None
    mul_313: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_64, sub_70);  div_64 = sub_70 = None
    mul_314: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_239, mul_270);  mul_270 = None
    sum_47: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_48: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_240: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_291, mul_313);  mul_291 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_819: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_240, [128, 4096])
    mm_126: "f32[128, 16384]" = torch.ops.aten.mm.default(view_819, permute_346);  permute_346 = None
    permute_347: "f32[4096, 128]" = torch.ops.aten.permute.default(view_819, [1, 0])
    mm_127: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_347, view_756);  view_756 = None
    permute_348: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_49: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_819, [0], True)
    view_820: "f32[4096]" = torch.ops.aten.reshape.default(sum_49, [4096]);  sum_49 = None
    permute_349: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_821: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_126, [1, 128, 16384]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_315: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_821, mul_266);  mul_266 = None
    mul_316: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_821, add_213);  view_821 = add_213 = None
    mul_317: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
    sub_71: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_317);  mul_317 = None
    mul_318: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_315, sub_71);  mul_315 = sub_71 = None
    mul_319: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_318, 0.7978845608028654);  mul_318 = None
    mul_320: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_319, 0.044715)
    pow_30: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_755, 2.0);  view_755 = None
    mul_321: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
    mul_322: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_320, mul_321);  mul_320 = mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_241: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_319, mul_322);  mul_319 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_323: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_316, 0.5);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_242: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_241, mul_323);  add_241 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_822: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_242, [128, 16384]);  add_242 = None
    mm_128: "f32[128, 4096]" = torch.ops.aten.mm.default(view_822, permute_350);  permute_350 = None
    permute_351: "f32[16384, 128]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_129: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_351, view_730);  permute_351 = None
    permute_352: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_50: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[16384]" = torch.ops.aten.reshape.default(sum_50, [16384]);  sum_50 = None
    permute_353: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_824: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_128, [1, 128, 4096]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_130: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_347, view_752);  permute_347 = view_752 = None
    permute_355: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[128, 4096]" = torch.ops.aten.mm.default(view_819, permute_356);  view_819 = permute_356 = None
    view_826: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 4096]);  mm_131 = None
    permute_357: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_827: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_826, [1, 128, 16, 256]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_358: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_827, [0, 2, 1, 3]);  view_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_828: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_358, [16, 128, 256]);  permute_358 = None
    bmm_60: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_359, view_828);  permute_359 = None
    bmm_61: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_828, permute_360);  view_828 = permute_360 = None
    view_829: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 256]);  bmm_60 = None
    view_830: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_324: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_830, alias_63);  view_830 = None
    sum_51: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [-1], True)
    mul_325: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_63, sum_51);  alias_63 = sum_51 = None
    sub_72: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_324, mul_325);  mul_324 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_65: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_72, primals_366);  sub_72 = primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_37: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1296, div_65, full_default_29);  slice_1296 = div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_831: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_37, [16, 128, 128]);  where_37 = None
    bmm_62: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_361, view_831);  permute_361 = None
    bmm_63: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_831, permute_362);  view_831 = permute_362 = None
    view_832: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_62, [1, 16, 256, 128]);  bmm_62 = None
    view_833: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_63, [1, 16, 128, 256]);  bmm_63 = None
    permute_363: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_832, [0, 1, 3, 2]);  view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_364: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_365: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_363, [0, 2, 1, 3]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1349: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_364, 3, 0, 64)
    slice_1350: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_364, 3, 64, 256);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1351: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_365, 3, 0, 64)
    slice_1352: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_365, 3, 64, 256);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_326: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1349, view_739)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_834: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_326, [1, 128, 16, 32, 2]);  mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_4: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_834, 4, 0)
    select_5: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_834, 4, 1);  view_834 = None
    neg_60: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_4);  select_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_32: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_60, 3, 1, 9223372036854775807, 2);  neg_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_36: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_243: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_36);  slice_scatter_32 = slice_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_327: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1349, view_740);  slice_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_244: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_243, mul_327);  add_243 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_328: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1351, view_739);  view_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_835: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_328, [1, 128, 16, 32, 2]);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_6: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_835, 4, 0)
    select_7: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_835, 4, 1);  view_835 = None
    neg_61: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_6);  select_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_40: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_44: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_245: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_44);  slice_scatter_40 = slice_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_329: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1351, view_740);  slice_1351 = view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_246: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_245, mul_329);  add_245 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_48: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1350, 3, 64, 9223372036854775807);  slice_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_52: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_244, 3, 0, 64);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_247: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_52);  slice_scatter_48 = slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_56: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1352, 3, 64, 9223372036854775807);  slice_1352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_60: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_246, 3, 0, 64);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_248: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_60);  slice_scatter_56 = slice_scatter_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_366: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_829, [0, 2, 1, 3]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_228: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    view_836: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_228, [1, 128, 4096]);  clone_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_837: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_248, [1, 128, 4096]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_838: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_247, [1, 128, 4096]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_839: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_836, [128, 4096]);  view_836 = None
    permute_367: "f32[4096, 128]" = torch.ops.aten.permute.default(view_839, [1, 0])
    mm_132: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_367, view_730);  permute_367 = None
    permute_368: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    mm_133: "f32[128, 4096]" = torch.ops.aten.mm.default(view_839, permute_369);  view_839 = permute_369 = None
    view_840: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 4096]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_249: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_824, view_840);  view_824 = view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_370: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_841: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_837, [128, 4096]);  view_837 = None
    permute_371: "f32[4096, 128]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_134: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_371, view_730);  permute_371 = None
    permute_372: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    mm_135: "f32[128, 4096]" = torch.ops.aten.mm.default(view_841, permute_373);  view_841 = permute_373 = None
    view_842: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 4096]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_250: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_249, view_842);  add_249 = view_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_374: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_843: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_838, [128, 4096]);  view_838 = None
    permute_375: "f32[4096, 128]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_136: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_375, view_730);  permute_375 = view_730 = None
    permute_376: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[128, 4096]" = torch.ops.aten.mm.default(view_843, permute_377);  view_843 = permute_377 = None
    view_844: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 4096]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_251: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_250, view_844);  add_250 = view_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_378: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_331: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_251, primals_262);  primals_262 = None
    mul_332: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_331, 4096)
    sum_52: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True)
    mul_333: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_331, mul_260);  mul_331 = None
    sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True);  mul_333 = None
    mul_334: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, sum_53);  sum_53 = None
    sub_74: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_332, sum_52);  mul_332 = sum_52 = None
    sub_75: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_74, mul_334);  sub_74 = mul_334 = None
    mul_335: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_66, sub_75);  div_66 = sub_75 = None
    mul_336: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_251, mul_260);  mul_260 = None
    sum_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 1]);  mul_336 = None
    sum_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_252: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_240, mul_335);  add_240 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_845: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_252, [128, 4096])
    mm_138: "f32[128, 16384]" = torch.ops.aten.mm.default(view_845, permute_379);  permute_379 = None
    permute_380: "f32[4096, 128]" = torch.ops.aten.permute.default(view_845, [1, 0])
    mm_139: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_380, view_728);  view_728 = None
    permute_381: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_56: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_845, [0], True)
    view_846: "f32[4096]" = torch.ops.aten.reshape.default(sum_56, [4096]);  sum_56 = None
    permute_382: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_847: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_138, [1, 128, 16384]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_337: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_847, mul_256);  mul_256 = None
    mul_338: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_847, add_205);  view_847 = add_205 = None
    mul_339: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
    sub_76: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_339);  mul_339 = None
    mul_340: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_337, sub_76);  mul_337 = sub_76 = None
    mul_341: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_340, 0.7978845608028654);  mul_340 = None
    mul_342: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_341, 0.044715)
    pow_31: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_727, 2.0);  view_727 = None
    mul_343: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
    mul_344: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_253: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_341, mul_344);  mul_341 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_345: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_338, 0.5);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_254: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_253, mul_345);  add_253 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_848: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_254, [128, 16384]);  add_254 = None
    mm_140: "f32[128, 4096]" = torch.ops.aten.mm.default(view_848, permute_383);  permute_383 = None
    permute_384: "f32[16384, 128]" = torch.ops.aten.permute.default(view_848, [1, 0])
    mm_141: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_384, view_702);  permute_384 = None
    permute_385: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_57: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True);  view_848 = None
    view_849: "f32[16384]" = torch.ops.aten.reshape.default(sum_57, [16384]);  sum_57 = None
    permute_386: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_850: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_140, [1, 128, 4096]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_142: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_380, view_724);  permute_380 = view_724 = None
    permute_388: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    mm_143: "f32[128, 4096]" = torch.ops.aten.mm.default(view_845, permute_389);  view_845 = permute_389 = None
    view_852: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None
    permute_390: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_853: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_852, [1, 128, 16, 256]);  view_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_391: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_854: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_391, [16, 128, 256]);  permute_391 = None
    bmm_64: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_392, view_854);  permute_392 = None
    bmm_65: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_854, permute_393);  view_854 = permute_393 = None
    view_855: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 256]);  bmm_64 = None
    view_856: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_346: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_856, alias_65);  view_856 = None
    sum_58: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
    mul_347: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_65, sum_58);  alias_65 = sum_58 = None
    sub_77: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_67: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_77, primals_363);  sub_77 = primals_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_38: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1248, div_67, full_default_29);  slice_1248 = div_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_857: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_38, [16, 128, 128]);  where_38 = None
    bmm_66: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_394, view_857);  permute_394 = None
    bmm_67: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_857, permute_395);  view_857 = permute_395 = None
    view_858: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_66, [1, 16, 256, 128]);  bmm_66 = None
    view_859: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_67, [1, 16, 128, 256]);  bmm_67 = None
    permute_396: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_858, [0, 1, 3, 2]);  view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_397: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_859, [0, 2, 1, 3]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_398: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_396, [0, 2, 1, 3]);  permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1353: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_397, 3, 0, 64)
    slice_1354: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_397, 3, 64, 256);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1355: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_398, 3, 0, 64)
    slice_1356: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_398, 3, 64, 256);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_348: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1353, view_711)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_860: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_348, [1, 128, 16, 32, 2]);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_8: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_860, 4, 0)
    select_9: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_860, 4, 1);  view_860 = None
    neg_62: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_8);  select_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_64: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_68: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_255: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_68);  slice_scatter_64 = slice_scatter_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_349: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1353, view_712);  slice_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_256: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_255, mul_349);  add_255 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_350: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1355, view_711);  view_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_861: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_350, [1, 128, 16, 32, 2]);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_10: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_861, 4, 0)
    select_11: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_861, 4, 1);  view_861 = None
    neg_63: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_10);  select_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_72: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_63, 3, 1, 9223372036854775807, 2);  neg_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_76: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_257: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_76);  slice_scatter_72 = slice_scatter_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_351: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1355, view_712);  slice_1355 = view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_258: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_257, mul_351);  add_257 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_80: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1354, 3, 64, 9223372036854775807);  slice_1354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_84: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_256, 3, 0, 64);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_259: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_84);  slice_scatter_80 = slice_scatter_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_88: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1356, 3, 64, 9223372036854775807);  slice_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_92: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_258, 3, 0, 64);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_260: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_92);  slice_scatter_88 = slice_scatter_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_399: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_855, [0, 2, 1, 3]);  view_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_229: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
    view_862: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_229, [1, 128, 4096]);  clone_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_863: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_260, [1, 128, 4096]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_864: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_259, [1, 128, 4096]);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_865: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_862, [128, 4096]);  view_862 = None
    permute_400: "f32[4096, 128]" = torch.ops.aten.permute.default(view_865, [1, 0])
    mm_144: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_400, view_702);  permute_400 = None
    permute_401: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[128, 4096]" = torch.ops.aten.mm.default(view_865, permute_402);  view_865 = permute_402 = None
    view_866: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_145, [1, 128, 4096]);  mm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_261: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_850, view_866);  view_850 = view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_403: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_867: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_863, [128, 4096]);  view_863 = None
    permute_404: "f32[4096, 128]" = torch.ops.aten.permute.default(view_867, [1, 0])
    mm_146: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_404, view_702);  permute_404 = None
    permute_405: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    mm_147: "f32[128, 4096]" = torch.ops.aten.mm.default(view_867, permute_406);  view_867 = permute_406 = None
    view_868: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_147, [1, 128, 4096]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_262: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_261, view_868);  add_261 = view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_407: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_869: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_864, [128, 4096]);  view_864 = None
    permute_408: "f32[4096, 128]" = torch.ops.aten.permute.default(view_869, [1, 0])
    mm_148: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_408, view_702);  permute_408 = view_702 = None
    permute_409: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    mm_149: "f32[128, 4096]" = torch.ops.aten.mm.default(view_869, permute_410);  view_869 = permute_410 = None
    view_870: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_149, [1, 128, 4096]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_263: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_262, view_870);  add_262 = view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_411: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_353: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_263, primals_252);  primals_252 = None
    mul_354: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_353, 4096)
    sum_59: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_353, mul_250);  mul_353 = None
    sum_60: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, sum_60);  sum_60 = None
    sub_79: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_354, sum_59);  mul_354 = sum_59 = None
    sub_80: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_79, mul_356);  sub_79 = mul_356 = None
    mul_357: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_68, sub_80);  div_68 = sub_80 = None
    mul_358: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_263, mul_250);  mul_250 = None
    sum_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_264: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_252, mul_357);  add_252 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_871: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_264, [128, 4096])
    mm_150: "f32[128, 16384]" = torch.ops.aten.mm.default(view_871, permute_412);  permute_412 = None
    permute_413: "f32[4096, 128]" = torch.ops.aten.permute.default(view_871, [1, 0])
    mm_151: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_413, view_700);  view_700 = None
    permute_414: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True)
    view_872: "f32[4096]" = torch.ops.aten.reshape.default(sum_63, [4096]);  sum_63 = None
    permute_415: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    view_873: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_150, [1, 128, 16384]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_359: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_873, mul_246);  mul_246 = None
    mul_360: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_873, add_197);  view_873 = add_197 = None
    mul_361: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
    sub_81: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_361);  mul_361 = None
    mul_362: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_359, sub_81);  mul_359 = sub_81 = None
    mul_363: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_362, 0.7978845608028654);  mul_362 = None
    mul_364: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_363, 0.044715)
    pow_32: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_699, 2.0);  view_699 = None
    mul_365: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
    mul_366: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_265: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_363, mul_366);  mul_363 = mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_367: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_360, 0.5);  mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_266: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_265, mul_367);  add_265 = mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_874: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_266, [128, 16384]);  add_266 = None
    mm_152: "f32[128, 4096]" = torch.ops.aten.mm.default(view_874, permute_416);  permute_416 = None
    permute_417: "f32[16384, 128]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_153: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_417, view_674);  permute_417 = None
    permute_418: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_64: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[16384]" = torch.ops.aten.reshape.default(sum_64, [16384]);  sum_64 = None
    permute_419: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    view_876: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_152, [1, 128, 4096]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_154: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_413, view_696);  permute_413 = view_696 = None
    permute_421: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    mm_155: "f32[128, 4096]" = torch.ops.aten.mm.default(view_871, permute_422);  view_871 = permute_422 = None
    view_878: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_155, [1, 128, 4096]);  mm_155 = None
    permute_423: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_879: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_878, [1, 128, 16, 256]);  view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_424: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_879, [0, 2, 1, 3]);  view_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_880: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_424, [16, 128, 256]);  permute_424 = None
    bmm_68: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_425, view_880);  permute_425 = None
    bmm_69: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_880, permute_426);  view_880 = permute_426 = None
    view_881: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 256]);  bmm_68 = None
    view_882: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_368: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_882, alias_67);  view_882 = None
    sum_65: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [-1], True)
    mul_369: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_67, sum_65);  alias_67 = sum_65 = None
    sub_82: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_69: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_82, primals_360);  sub_82 = primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_39: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1200, div_69, full_default_29);  slice_1200 = div_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_883: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_39, [16, 128, 128]);  where_39 = None
    bmm_70: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_427, view_883);  permute_427 = None
    bmm_71: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_883, permute_428);  view_883 = permute_428 = None
    view_884: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_70, [1, 16, 256, 128]);  bmm_70 = None
    view_885: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_71, [1, 16, 128, 256]);  bmm_71 = None
    permute_429: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_884, [0, 1, 3, 2]);  view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_430: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_885, [0, 2, 1, 3]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_431: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_429, [0, 2, 1, 3]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1357: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_430, 3, 0, 64)
    slice_1358: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_430, 3, 64, 256);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1359: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_431, 3, 0, 64)
    slice_1360: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_431, 3, 64, 256);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_370: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1357, view_683)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_886: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_370, [1, 128, 16, 32, 2]);  mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_12: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_886, 4, 0)
    select_13: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_886, 4, 1);  view_886 = None
    neg_64: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_12);  select_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_96: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_100: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_267: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_100);  slice_scatter_96 = slice_scatter_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_371: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1357, view_684);  slice_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_268: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_267, mul_371);  add_267 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_372: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1359, view_683);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_887: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_372, [1, 128, 16, 32, 2]);  mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_14: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_887, 4, 0)
    select_15: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_887, 4, 1);  view_887 = None
    neg_65: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_14);  select_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_104: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_269: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_108);  slice_scatter_104 = slice_scatter_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_373: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1359, view_684);  slice_1359 = view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_270: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_269, mul_373);  add_269 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_112: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1358, 3, 64, 9223372036854775807);  slice_1358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_116: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_268, 3, 0, 64);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_271: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_116);  slice_scatter_112 = slice_scatter_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_120: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1360, 3, 64, 9223372036854775807);  slice_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_124: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_270, 3, 0, 64);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_272: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_124);  slice_scatter_120 = slice_scatter_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_432: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_881, [0, 2, 1, 3]);  view_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_230: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
    view_888: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_230, [1, 128, 4096]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_889: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_272, [1, 128, 4096]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_890: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_271, [1, 128, 4096]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_891: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_888, [128, 4096]);  view_888 = None
    permute_433: "f32[4096, 128]" = torch.ops.aten.permute.default(view_891, [1, 0])
    mm_156: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_433, view_674);  permute_433 = None
    permute_434: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    mm_157: "f32[128, 4096]" = torch.ops.aten.mm.default(view_891, permute_435);  view_891 = permute_435 = None
    view_892: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_157, [1, 128, 4096]);  mm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_273: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_876, view_892);  view_876 = view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_436: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_893: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_889, [128, 4096]);  view_889 = None
    permute_437: "f32[4096, 128]" = torch.ops.aten.permute.default(view_893, [1, 0])
    mm_158: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_437, view_674);  permute_437 = None
    permute_438: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    mm_159: "f32[128, 4096]" = torch.ops.aten.mm.default(view_893, permute_439);  view_893 = permute_439 = None
    view_894: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_159, [1, 128, 4096]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_274: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_273, view_894);  add_273 = view_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_440: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_895: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_890, [128, 4096]);  view_890 = None
    permute_441: "f32[4096, 128]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_160: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_441, view_674);  permute_441 = view_674 = None
    permute_442: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    mm_161: "f32[128, 4096]" = torch.ops.aten.mm.default(view_895, permute_443);  view_895 = permute_443 = None
    view_896: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_161, [1, 128, 4096]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_275: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_274, view_896);  add_274 = view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_444: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_375: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_275, primals_242);  primals_242 = None
    mul_376: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_375, 4096)
    sum_66: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True)
    mul_377: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_375, mul_240);  mul_375 = None
    sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
    mul_378: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, sum_67);  sum_67 = None
    sub_84: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_376, sum_66);  mul_376 = sum_66 = None
    sub_85: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_84, mul_378);  sub_84 = mul_378 = None
    mul_379: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_70, sub_85);  div_70 = sub_85 = None
    mul_380: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_275, mul_240);  mul_240 = None
    sum_68: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1]);  mul_380 = None
    sum_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_276: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_264, mul_379);  add_264 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_897: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_276, [128, 4096])
    mm_162: "f32[128, 16384]" = torch.ops.aten.mm.default(view_897, permute_445);  permute_445 = None
    permute_446: "f32[4096, 128]" = torch.ops.aten.permute.default(view_897, [1, 0])
    mm_163: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_446, view_672);  view_672 = None
    permute_447: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True)
    view_898: "f32[4096]" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
    permute_448: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    view_899: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_162, [1, 128, 16384]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_381: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_899, mul_236);  mul_236 = None
    mul_382: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_899, add_189);  view_899 = add_189 = None
    mul_383: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
    sub_86: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_383);  mul_383 = None
    mul_384: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_381, sub_86);  mul_381 = sub_86 = None
    mul_385: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_384, 0.7978845608028654);  mul_384 = None
    mul_386: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_385, 0.044715)
    pow_33: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_671, 2.0);  view_671 = None
    mul_387: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
    mul_388: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_277: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_385, mul_388);  mul_385 = mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_389: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_382, 0.5);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_278: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_277, mul_389);  add_277 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_900: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_278, [128, 16384]);  add_278 = None
    mm_164: "f32[128, 4096]" = torch.ops.aten.mm.default(view_900, permute_449);  permute_449 = None
    permute_450: "f32[16384, 128]" = torch.ops.aten.permute.default(view_900, [1, 0])
    mm_165: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_450, view_646);  permute_450 = None
    permute_451: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_71: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True);  view_900 = None
    view_901: "f32[16384]" = torch.ops.aten.reshape.default(sum_71, [16384]);  sum_71 = None
    permute_452: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
    view_902: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_164, [1, 128, 4096]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_166: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_446, view_668);  permute_446 = view_668 = None
    permute_454: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[128, 4096]" = torch.ops.aten.mm.default(view_897, permute_455);  view_897 = permute_455 = None
    view_904: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_167, [1, 128, 4096]);  mm_167 = None
    permute_456: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_905: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_904, [1, 128, 16, 256]);  view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_457: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_905, [0, 2, 1, 3]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_906: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_457, [16, 128, 256]);  permute_457 = None
    bmm_72: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_458, view_906);  permute_458 = None
    bmm_73: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_906, permute_459);  view_906 = permute_459 = None
    view_907: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_72, [1, 16, 128, 256]);  bmm_72 = None
    view_908: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_390: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_908, alias_69);  view_908 = None
    sum_72: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
    mul_391: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_69, sum_72);  alias_69 = sum_72 = None
    sub_87: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_71: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_87, primals_357);  sub_87 = primals_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_40: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1152, div_71, full_default_29);  slice_1152 = div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_909: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_40, [16, 128, 128]);  where_40 = None
    bmm_74: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_460, view_909);  permute_460 = None
    bmm_75: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_909, permute_461);  view_909 = permute_461 = None
    view_910: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_74, [1, 16, 256, 128]);  bmm_74 = None
    view_911: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_75, [1, 16, 128, 256]);  bmm_75 = None
    permute_462: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_910, [0, 1, 3, 2]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_463: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_911, [0, 2, 1, 3]);  view_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_464: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_462, [0, 2, 1, 3]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1361: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_463, 3, 0, 64)
    slice_1362: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_463, 3, 64, 256);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1363: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_464, 3, 0, 64)
    slice_1364: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_464, 3, 64, 256);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_392: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1361, view_655)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_912: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_392, [1, 128, 16, 32, 2]);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_16: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_912, 4, 0)
    select_17: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_912, 4, 1);  view_912 = None
    neg_66: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_16);  select_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_128: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_66, 3, 1, 9223372036854775807, 2);  neg_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_132: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_279: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_132);  slice_scatter_128 = slice_scatter_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_393: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1361, view_656);  slice_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_280: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_279, mul_393);  add_279 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_394: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1363, view_655);  view_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_913: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_394, [1, 128, 16, 32, 2]);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_18: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_913, 4, 0)
    select_19: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_913, 4, 1);  view_913 = None
    neg_67: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_18);  select_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_136: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_140: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_281: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_140);  slice_scatter_136 = slice_scatter_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_395: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1363, view_656);  slice_1363 = view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_282: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_281, mul_395);  add_281 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_144: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1362, 3, 64, 9223372036854775807);  slice_1362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_148: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_280, 3, 0, 64);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_283: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_148);  slice_scatter_144 = slice_scatter_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_152: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1364, 3, 64, 9223372036854775807);  slice_1364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_156: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_282, 3, 0, 64);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_284: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_156);  slice_scatter_152 = slice_scatter_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_465: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_907, [0, 2, 1, 3]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_231: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
    view_914: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_231, [1, 128, 4096]);  clone_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_915: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_284, [1, 128, 4096]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_916: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_283, [1, 128, 4096]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_917: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_914, [128, 4096]);  view_914 = None
    permute_466: "f32[4096, 128]" = torch.ops.aten.permute.default(view_917, [1, 0])
    mm_168: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_466, view_646);  permute_466 = None
    permute_467: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[128, 4096]" = torch.ops.aten.mm.default(view_917, permute_468);  view_917 = permute_468 = None
    view_918: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_169, [1, 128, 4096]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_285: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_902, view_918);  view_902 = view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_469: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_919: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_915, [128, 4096]);  view_915 = None
    permute_470: "f32[4096, 128]" = torch.ops.aten.permute.default(view_919, [1, 0])
    mm_170: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_470, view_646);  permute_470 = None
    permute_471: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_170, [1, 0]);  mm_170 = None
    mm_171: "f32[128, 4096]" = torch.ops.aten.mm.default(view_919, permute_472);  view_919 = permute_472 = None
    view_920: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_171, [1, 128, 4096]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_286: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_285, view_920);  add_285 = view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_473: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_921: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_916, [128, 4096]);  view_916 = None
    permute_474: "f32[4096, 128]" = torch.ops.aten.permute.default(view_921, [1, 0])
    mm_172: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_474, view_646);  permute_474 = view_646 = None
    permute_475: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    mm_173: "f32[128, 4096]" = torch.ops.aten.mm.default(view_921, permute_476);  view_921 = permute_476 = None
    view_922: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_173, [1, 128, 4096]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_287: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_286, view_922);  add_286 = view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_477: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_397: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_287, primals_232);  primals_232 = None
    mul_398: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_397, 4096)
    sum_73: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_397, mul_230);  mul_397 = None
    sum_74: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, sum_74);  sum_74 = None
    sub_89: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_398, sum_73);  mul_398 = sum_73 = None
    sub_90: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_89, mul_400);  sub_89 = mul_400 = None
    mul_401: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_72, sub_90);  div_72 = sub_90 = None
    mul_402: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_287, mul_230);  mul_230 = None
    sum_75: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_76: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_288: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_276, mul_401);  add_276 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_923: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_288, [128, 4096])
    mm_174: "f32[128, 16384]" = torch.ops.aten.mm.default(view_923, permute_478);  permute_478 = None
    permute_479: "f32[4096, 128]" = torch.ops.aten.permute.default(view_923, [1, 0])
    mm_175: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_479, view_644);  view_644 = None
    permute_480: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_77: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_923, [0], True)
    view_924: "f32[4096]" = torch.ops.aten.reshape.default(sum_77, [4096]);  sum_77 = None
    permute_481: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    view_925: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_174, [1, 128, 16384]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_403: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_925, mul_226);  mul_226 = None
    mul_404: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_925, add_181);  view_925 = add_181 = None
    mul_405: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
    sub_91: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
    mul_406: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_403, sub_91);  mul_403 = sub_91 = None
    mul_407: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
    mul_408: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_407, 0.044715)
    pow_34: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_643, 2.0);  view_643 = None
    mul_409: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
    mul_410: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_289: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_407, mul_410);  mul_407 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_411: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_404, 0.5);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_290: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_289, mul_411);  add_289 = mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_926: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_290, [128, 16384]);  add_290 = None
    mm_176: "f32[128, 4096]" = torch.ops.aten.mm.default(view_926, permute_482);  permute_482 = None
    permute_483: "f32[16384, 128]" = torch.ops.aten.permute.default(view_926, [1, 0])
    mm_177: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_483, view_618);  permute_483 = None
    permute_484: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_78: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True);  view_926 = None
    view_927: "f32[16384]" = torch.ops.aten.reshape.default(sum_78, [16384]);  sum_78 = None
    permute_485: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    view_928: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_176, [1, 128, 4096]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_178: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_479, view_640);  permute_479 = view_640 = None
    permute_487: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    mm_179: "f32[128, 4096]" = torch.ops.aten.mm.default(view_923, permute_488);  view_923 = permute_488 = None
    view_930: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_179, [1, 128, 4096]);  mm_179 = None
    permute_489: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_931: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_930, [1, 128, 16, 256]);  view_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_490: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_932: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_490, [16, 128, 256]);  permute_490 = None
    bmm_76: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_491, view_932);  permute_491 = None
    bmm_77: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_932, permute_492);  view_932 = permute_492 = None
    view_933: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_76, [1, 16, 128, 256]);  bmm_76 = None
    view_934: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_412: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_934, alias_71);  view_934 = None
    sum_79: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
    mul_413: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_71, sum_79);  alias_71 = sum_79 = None
    sub_92: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_73: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_92, primals_354);  sub_92 = primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_41: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1104, div_73, full_default_29);  slice_1104 = div_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_935: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_41, [16, 128, 128]);  where_41 = None
    bmm_78: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_493, view_935);  permute_493 = None
    bmm_79: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_935, permute_494);  view_935 = permute_494 = None
    view_936: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_78, [1, 16, 256, 128]);  bmm_78 = None
    view_937: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_79, [1, 16, 128, 256]);  bmm_79 = None
    permute_495: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_936, [0, 1, 3, 2]);  view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_496: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_937, [0, 2, 1, 3]);  view_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_497: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_495, [0, 2, 1, 3]);  permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1365: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_496, 3, 0, 64)
    slice_1366: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_496, 3, 64, 256);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1367: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_497, 3, 0, 64)
    slice_1368: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_497, 3, 64, 256);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_414: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1365, view_627)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_938: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_414, [1, 128, 16, 32, 2]);  mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_20: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_938, 4, 0)
    select_21: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_938, 4, 1);  view_938 = None
    neg_68: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_20);  select_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_160: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_164: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_291: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_164);  slice_scatter_160 = slice_scatter_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_415: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1365, view_628);  slice_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_292: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_291, mul_415);  add_291 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_416: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1367, view_627);  view_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_939: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_416, [1, 128, 16, 32, 2]);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_22: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_939, 4, 0)
    select_23: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_939, 4, 1);  view_939 = None
    neg_69: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_22);  select_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_168: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_69, 3, 1, 9223372036854775807, 2);  neg_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_172: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_293: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_172);  slice_scatter_168 = slice_scatter_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_417: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1367, view_628);  slice_1367 = view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_294: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_293, mul_417);  add_293 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_176: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1366, 3, 64, 9223372036854775807);  slice_1366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_180: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_292, 3, 0, 64);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_295: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_180);  slice_scatter_176 = slice_scatter_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_184: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1368, 3, 64, 9223372036854775807);  slice_1368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_188: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_294, 3, 0, 64);  add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_296: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_188);  slice_scatter_184 = slice_scatter_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_498: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_933, [0, 2, 1, 3]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_232: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
    view_940: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_232, [1, 128, 4096]);  clone_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_941: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_296, [1, 128, 4096]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_942: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_295, [1, 128, 4096]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_943: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_940, [128, 4096]);  view_940 = None
    permute_499: "f32[4096, 128]" = torch.ops.aten.permute.default(view_943, [1, 0])
    mm_180: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_499, view_618);  permute_499 = None
    permute_500: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    mm_181: "f32[128, 4096]" = torch.ops.aten.mm.default(view_943, permute_501);  view_943 = permute_501 = None
    view_944: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_181, [1, 128, 4096]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_297: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_928, view_944);  view_928 = view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_502: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_945: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_941, [128, 4096]);  view_941 = None
    permute_503: "f32[4096, 128]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_182: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_503, view_618);  permute_503 = None
    permute_504: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_182, [1, 0]);  mm_182 = None
    mm_183: "f32[128, 4096]" = torch.ops.aten.mm.default(view_945, permute_505);  view_945 = permute_505 = None
    view_946: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_298: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_297, view_946);  add_297 = view_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_506: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_947: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_942, [128, 4096]);  view_942 = None
    permute_507: "f32[4096, 128]" = torch.ops.aten.permute.default(view_947, [1, 0])
    mm_184: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_507, view_618);  permute_507 = view_618 = None
    permute_508: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    mm_185: "f32[128, 4096]" = torch.ops.aten.mm.default(view_947, permute_509);  view_947 = permute_509 = None
    view_948: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_185, [1, 128, 4096]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_299: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_298, view_948);  add_298 = view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_510: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_419: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_299, primals_222);  primals_222 = None
    mul_420: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_419, 4096)
    sum_80: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
    mul_421: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_419, mul_220);  mul_419 = None
    sum_81: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    mul_422: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, sum_81);  sum_81 = None
    sub_94: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_420, sum_80);  mul_420 = sum_80 = None
    sub_95: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_94, mul_422);  sub_94 = mul_422 = None
    mul_423: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_74, sub_95);  div_74 = sub_95 = None
    mul_424: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_299, mul_220);  mul_220 = None
    sum_82: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
    sum_83: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_300: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_288, mul_423);  add_288 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_949: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_300, [128, 4096])
    mm_186: "f32[128, 16384]" = torch.ops.aten.mm.default(view_949, permute_511);  permute_511 = None
    permute_512: "f32[4096, 128]" = torch.ops.aten.permute.default(view_949, [1, 0])
    mm_187: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_512, view_616);  view_616 = None
    permute_513: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_84: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_949, [0], True)
    view_950: "f32[4096]" = torch.ops.aten.reshape.default(sum_84, [4096]);  sum_84 = None
    permute_514: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_513, [1, 0]);  permute_513 = None
    view_951: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_186, [1, 128, 16384]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_425: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_951, mul_216);  mul_216 = None
    mul_426: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_951, add_173);  view_951 = add_173 = None
    mul_427: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
    sub_96: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_427);  mul_427 = None
    mul_428: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_425, sub_96);  mul_425 = sub_96 = None
    mul_429: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_428, 0.7978845608028654);  mul_428 = None
    mul_430: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_429, 0.044715)
    pow_35: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_615, 2.0);  view_615 = None
    mul_431: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
    mul_432: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_301: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_429, mul_432);  mul_429 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_433: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_426, 0.5);  mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_302: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_301, mul_433);  add_301 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_952: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_302, [128, 16384]);  add_302 = None
    mm_188: "f32[128, 4096]" = torch.ops.aten.mm.default(view_952, permute_515);  permute_515 = None
    permute_516: "f32[16384, 128]" = torch.ops.aten.permute.default(view_952, [1, 0])
    mm_189: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_516, view_590);  permute_516 = None
    permute_517: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_85: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True);  view_952 = None
    view_953: "f32[16384]" = torch.ops.aten.reshape.default(sum_85, [16384]);  sum_85 = None
    permute_518: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_517, [1, 0]);  permute_517 = None
    view_954: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_188, [1, 128, 4096]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_190: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_512, view_612);  permute_512 = view_612 = None
    permute_520: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    mm_191: "f32[128, 4096]" = torch.ops.aten.mm.default(view_949, permute_521);  view_949 = permute_521 = None
    view_956: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_191, [1, 128, 4096]);  mm_191 = None
    permute_522: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_957: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_956, [1, 128, 16, 256]);  view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_523: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_957, [0, 2, 1, 3]);  view_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_958: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_523, [16, 128, 256]);  permute_523 = None
    bmm_80: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_524, view_958);  permute_524 = None
    bmm_81: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_958, permute_525);  view_958 = permute_525 = None
    view_959: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_80, [1, 16, 128, 256]);  bmm_80 = None
    view_960: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_434: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_960, alias_73);  view_960 = None
    sum_86: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [-1], True)
    mul_435: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_73, sum_86);  alias_73 = sum_86 = None
    sub_97: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_75: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_97, primals_351);  sub_97 = primals_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_42: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1056, div_75, full_default_29);  slice_1056 = div_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_961: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_42, [16, 128, 128]);  where_42 = None
    bmm_82: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_526, view_961);  permute_526 = None
    bmm_83: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_961, permute_527);  view_961 = permute_527 = None
    view_962: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_82, [1, 16, 256, 128]);  bmm_82 = None
    view_963: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_83, [1, 16, 128, 256]);  bmm_83 = None
    permute_528: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_962, [0, 1, 3, 2]);  view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_529: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_963, [0, 2, 1, 3]);  view_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_530: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_528, [0, 2, 1, 3]);  permute_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1369: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_529, 3, 0, 64)
    slice_1370: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_529, 3, 64, 256);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1371: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_530, 3, 0, 64)
    slice_1372: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_530, 3, 64, 256);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_436: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1369, view_599)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_964: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_436, [1, 128, 16, 32, 2]);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_24: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_964, 4, 0)
    select_25: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_964, 4, 1);  view_964 = None
    neg_70: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_24);  select_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_192: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_196: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_303: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_196);  slice_scatter_192 = slice_scatter_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_437: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1369, view_600);  slice_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_304: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_303, mul_437);  add_303 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_438: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1371, view_599);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_965: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_438, [1, 128, 16, 32, 2]);  mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_26: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_965, 4, 0)
    select_27: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_965, 4, 1);  view_965 = None
    neg_71: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_26);  select_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_200: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_305: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_204);  slice_scatter_200 = slice_scatter_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_439: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1371, view_600);  slice_1371 = view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_306: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_305, mul_439);  add_305 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_208: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1370, 3, 64, 9223372036854775807);  slice_1370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_212: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_304, 3, 0, 64);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_307: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_212);  slice_scatter_208 = slice_scatter_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_216: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1372, 3, 64, 9223372036854775807);  slice_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_220: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_306, 3, 0, 64);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_308: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_220);  slice_scatter_216 = slice_scatter_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_531: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_959, [0, 2, 1, 3]);  view_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_233: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
    view_966: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_233, [1, 128, 4096]);  clone_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_967: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_308, [1, 128, 4096]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_968: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_307, [1, 128, 4096]);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_969: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_966, [128, 4096]);  view_966 = None
    permute_532: "f32[4096, 128]" = torch.ops.aten.permute.default(view_969, [1, 0])
    mm_192: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_532, view_590);  permute_532 = None
    permute_533: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    mm_193: "f32[128, 4096]" = torch.ops.aten.mm.default(view_969, permute_534);  view_969 = permute_534 = None
    view_970: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_193, [1, 128, 4096]);  mm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_309: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_954, view_970);  view_954 = view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_535: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_971: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_967, [128, 4096]);  view_967 = None
    permute_536: "f32[4096, 128]" = torch.ops.aten.permute.default(view_971, [1, 0])
    mm_194: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_536, view_590);  permute_536 = None
    permute_537: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_194, [1, 0]);  mm_194 = None
    mm_195: "f32[128, 4096]" = torch.ops.aten.mm.default(view_971, permute_538);  view_971 = permute_538 = None
    view_972: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_195, [1, 128, 4096]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_310: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_309, view_972);  add_309 = view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_539: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_973: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_968, [128, 4096]);  view_968 = None
    permute_540: "f32[4096, 128]" = torch.ops.aten.permute.default(view_973, [1, 0])
    mm_196: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_540, view_590);  permute_540 = view_590 = None
    permute_541: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    mm_197: "f32[128, 4096]" = torch.ops.aten.mm.default(view_973, permute_542);  view_973 = permute_542 = None
    view_974: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_197, [1, 128, 4096]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_311: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_310, view_974);  add_310 = view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_543: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_441: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_311, primals_212);  primals_212 = None
    mul_442: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_441, 4096)
    sum_87: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
    mul_443: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_441, mul_210);  mul_441 = None
    sum_88: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
    mul_444: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, sum_88);  sum_88 = None
    sub_99: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_442, sum_87);  mul_442 = sum_87 = None
    sub_100: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_99, mul_444);  sub_99 = mul_444 = None
    mul_445: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_76, sub_100);  div_76 = sub_100 = None
    mul_446: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_311, mul_210);  mul_210 = None
    sum_89: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
    sum_90: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_312: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_300, mul_445);  add_300 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_975: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_312, [128, 4096])
    mm_198: "f32[128, 16384]" = torch.ops.aten.mm.default(view_975, permute_544);  permute_544 = None
    permute_545: "f32[4096, 128]" = torch.ops.aten.permute.default(view_975, [1, 0])
    mm_199: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_545, view_588);  view_588 = None
    permute_546: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_91: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True)
    view_976: "f32[4096]" = torch.ops.aten.reshape.default(sum_91, [4096]);  sum_91 = None
    permute_547: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_977: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_198, [1, 128, 16384]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_447: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_977, mul_206);  mul_206 = None
    mul_448: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_977, add_165);  view_977 = add_165 = None
    mul_449: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
    sub_101: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_449);  mul_449 = None
    mul_450: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_447, sub_101);  mul_447 = sub_101 = None
    mul_451: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_450, 0.7978845608028654);  mul_450 = None
    mul_452: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_451, 0.044715)
    pow_36: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_587, 2.0);  view_587 = None
    mul_453: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
    mul_454: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_313: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_451, mul_454);  mul_451 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_455: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_448, 0.5);  mul_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_314: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_313, mul_455);  add_313 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_978: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_314, [128, 16384]);  add_314 = None
    mm_200: "f32[128, 4096]" = torch.ops.aten.mm.default(view_978, permute_548);  permute_548 = None
    permute_549: "f32[16384, 128]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_201: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_549, view_562);  permute_549 = None
    permute_550: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_92: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
    view_979: "f32[16384]" = torch.ops.aten.reshape.default(sum_92, [16384]);  sum_92 = None
    permute_551: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_980: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_200, [1, 128, 4096]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_202: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_545, view_584);  permute_545 = view_584 = None
    permute_553: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    mm_203: "f32[128, 4096]" = torch.ops.aten.mm.default(view_975, permute_554);  view_975 = permute_554 = None
    view_982: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None
    permute_555: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_983: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_982, [1, 128, 16, 256]);  view_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_556: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_983, [0, 2, 1, 3]);  view_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_984: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_556, [16, 128, 256]);  permute_556 = None
    bmm_84: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_557, view_984);  permute_557 = None
    bmm_85: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_984, permute_558);  view_984 = permute_558 = None
    view_985: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_84, [1, 16, 128, 256]);  bmm_84 = None
    view_986: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_456: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_986, alias_75);  view_986 = None
    sum_93: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [-1], True)
    mul_457: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_75, sum_93);  alias_75 = sum_93 = None
    sub_102: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_77: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_102, primals_348);  sub_102 = primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_43: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1008, div_77, full_default_29);  slice_1008 = div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_987: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_43, [16, 128, 128]);  where_43 = None
    bmm_86: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_559, view_987);  permute_559 = None
    bmm_87: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_987, permute_560);  view_987 = permute_560 = None
    view_988: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_86, [1, 16, 256, 128]);  bmm_86 = None
    view_989: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_87, [1, 16, 128, 256]);  bmm_87 = None
    permute_561: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_988, [0, 1, 3, 2]);  view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_562: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_989, [0, 2, 1, 3]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_563: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_561, [0, 2, 1, 3]);  permute_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1373: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_562, 3, 0, 64)
    slice_1374: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_562, 3, 64, 256);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1375: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_563, 3, 0, 64)
    slice_1376: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_563, 3, 64, 256);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_458: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1373, view_571)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_990: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_458, [1, 128, 16, 32, 2]);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_28: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_990, 4, 0)
    select_29: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_990, 4, 1);  view_990 = None
    neg_72: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_28);  select_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_224: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_72, 3, 1, 9223372036854775807, 2);  neg_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_228: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_315: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_224, slice_scatter_228);  slice_scatter_224 = slice_scatter_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_459: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1373, view_572);  slice_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_316: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_315, mul_459);  add_315 = mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_460: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1375, view_571);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_991: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_460, [1, 128, 16, 32, 2]);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_30: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_991, 4, 0)
    select_31: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_991, 4, 1);  view_991 = None
    neg_73: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_30);  select_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_232: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_236: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_317: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_232, slice_scatter_236);  slice_scatter_232 = slice_scatter_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_461: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1375, view_572);  slice_1375 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_318: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_317, mul_461);  add_317 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_240: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1374, 3, 64, 9223372036854775807);  slice_1374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_244: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_316, 3, 0, 64);  add_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_319: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_240, slice_scatter_244);  slice_scatter_240 = slice_scatter_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_248: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1376, 3, 64, 9223372036854775807);  slice_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_252: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_318, 3, 0, 64);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_320: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_248, slice_scatter_252);  slice_scatter_248 = slice_scatter_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_564: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_234: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
    view_992: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_234, [1, 128, 4096]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_993: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_320, [1, 128, 4096]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_994: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_319, [1, 128, 4096]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_995: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_992, [128, 4096]);  view_992 = None
    permute_565: "f32[4096, 128]" = torch.ops.aten.permute.default(view_995, [1, 0])
    mm_204: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_565, view_562);  permute_565 = None
    permute_566: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_204, [1, 0]);  mm_204 = None
    mm_205: "f32[128, 4096]" = torch.ops.aten.mm.default(view_995, permute_567);  view_995 = permute_567 = None
    view_996: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_205, [1, 128, 4096]);  mm_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_321: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_980, view_996);  view_980 = view_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_568: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_566, [1, 0]);  permute_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_997: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_993, [128, 4096]);  view_993 = None
    permute_569: "f32[4096, 128]" = torch.ops.aten.permute.default(view_997, [1, 0])
    mm_206: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_569, view_562);  permute_569 = None
    permute_570: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_206, [1, 0]);  mm_206 = None
    mm_207: "f32[128, 4096]" = torch.ops.aten.mm.default(view_997, permute_571);  view_997 = permute_571 = None
    view_998: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_207, [1, 128, 4096]);  mm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_322: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_321, view_998);  add_321 = view_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_572: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_570, [1, 0]);  permute_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_999: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_994, [128, 4096]);  view_994 = None
    permute_573: "f32[4096, 128]" = torch.ops.aten.permute.default(view_999, [1, 0])
    mm_208: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_573, view_562);  permute_573 = view_562 = None
    permute_574: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    mm_209: "f32[128, 4096]" = torch.ops.aten.mm.default(view_999, permute_575);  view_999 = permute_575 = None
    view_1000: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_209, [1, 128, 4096]);  mm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_323: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_322, view_1000);  add_322 = view_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_576: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_463: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_323, primals_202);  primals_202 = None
    mul_464: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_463, 4096)
    sum_94: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True)
    mul_465: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_463, mul_200);  mul_463 = None
    sum_95: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [2], True);  mul_465 = None
    mul_466: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, sum_95);  sum_95 = None
    sub_104: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_464, sum_94);  mul_464 = sum_94 = None
    sub_105: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_104, mul_466);  sub_104 = mul_466 = None
    mul_467: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_78, sub_105);  div_78 = sub_105 = None
    mul_468: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_323, mul_200);  mul_200 = None
    sum_96: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 1]);  mul_468 = None
    sum_97: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_324: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_312, mul_467);  add_312 = mul_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1001: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_324, [128, 4096])
    mm_210: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1001, permute_577);  permute_577 = None
    permute_578: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1001, [1, 0])
    mm_211: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_578, view_560);  view_560 = None
    permute_579: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_98: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1001, [0], True)
    view_1002: "f32[4096]" = torch.ops.aten.reshape.default(sum_98, [4096]);  sum_98 = None
    permute_580: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_1003: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_210, [1, 128, 16384]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_469: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1003, mul_196);  mul_196 = None
    mul_470: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1003, add_157);  view_1003 = add_157 = None
    mul_471: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
    sub_106: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_471);  mul_471 = None
    mul_472: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_469, sub_106);  mul_469 = sub_106 = None
    mul_473: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_472, 0.7978845608028654);  mul_472 = None
    mul_474: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_473, 0.044715)
    pow_37: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_559, 2.0);  view_559 = None
    mul_475: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
    mul_476: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_325: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_473, mul_476);  mul_473 = mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_477: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_470, 0.5);  mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_326: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_325, mul_477);  add_325 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1004: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_326, [128, 16384]);  add_326 = None
    mm_212: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1004, permute_581);  permute_581 = None
    permute_582: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1004, [1, 0])
    mm_213: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_582, view_534);  permute_582 = None
    permute_583: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_99: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
    view_1005: "f32[16384]" = torch.ops.aten.reshape.default(sum_99, [16384]);  sum_99 = None
    permute_584: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_583, [1, 0]);  permute_583 = None
    view_1006: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_212, [1, 128, 4096]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_214: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_578, view_556);  permute_578 = view_556 = None
    permute_586: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_214, [1, 0]);  mm_214 = None
    mm_215: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1001, permute_587);  view_1001 = permute_587 = None
    view_1008: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_215, [1, 128, 4096]);  mm_215 = None
    permute_588: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1009: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1008, [1, 128, 16, 256]);  view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_589: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1009, [0, 2, 1, 3]);  view_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1010: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_589, [16, 128, 256]);  permute_589 = None
    bmm_88: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_590, view_1010);  permute_590 = None
    bmm_89: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1010, permute_591);  view_1010 = permute_591 = None
    view_1011: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_88, [1, 16, 128, 256]);  bmm_88 = None
    view_1012: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_478: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1012, alias_77);  view_1012 = None
    sum_100: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
    mul_479: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_77, sum_100);  alias_77 = sum_100 = None
    sub_107: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_79: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_107, primals_345);  sub_107 = primals_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_44: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_960, div_79, full_default_29);  slice_960 = div_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1013: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_44, [16, 128, 128]);  where_44 = None
    bmm_90: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_592, view_1013);  permute_592 = None
    bmm_91: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1013, permute_593);  view_1013 = permute_593 = None
    view_1014: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_90, [1, 16, 256, 128]);  bmm_90 = None
    view_1015: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_91, [1, 16, 128, 256]);  bmm_91 = None
    permute_594: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1014, [0, 1, 3, 2]);  view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_595: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1015, [0, 2, 1, 3]);  view_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_596: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_594, [0, 2, 1, 3]);  permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1377: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_595, 3, 0, 64)
    slice_1378: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_595, 3, 64, 256);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1379: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_596, 3, 0, 64)
    slice_1380: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_596, 3, 64, 256);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_480: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1377, view_543)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1016: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_480, [1, 128, 16, 32, 2]);  mul_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_32: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1016, 4, 0)
    select_33: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1016, 4, 1);  view_1016 = None
    neg_74: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_32);  select_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_256: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_260: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_327: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_256, slice_scatter_260);  slice_scatter_256 = slice_scatter_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_481: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1377, view_544);  slice_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_328: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_327, mul_481);  add_327 = mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_482: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1379, view_543);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1017: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_482, [1, 128, 16, 32, 2]);  mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_34: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1017, 4, 0)
    select_35: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1017, 4, 1);  view_1017 = None
    neg_75: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_34);  select_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_264: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_75, 3, 1, 9223372036854775807, 2);  neg_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_268: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_329: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_264, slice_scatter_268);  slice_scatter_264 = slice_scatter_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_483: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1379, view_544);  slice_1379 = view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_330: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_329, mul_483);  add_329 = mul_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_272: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1378, 3, 64, 9223372036854775807);  slice_1378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_276: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_328, 3, 0, 64);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_331: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_272, slice_scatter_276);  slice_scatter_272 = slice_scatter_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_280: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1380, 3, 64, 9223372036854775807);  slice_1380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_284: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_330, 3, 0, 64);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_332: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_280, slice_scatter_284);  slice_scatter_280 = slice_scatter_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_597: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1011, [0, 2, 1, 3]);  view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_235: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
    view_1018: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_235, [1, 128, 4096]);  clone_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1019: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_332, [1, 128, 4096]);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1020: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_331, [1, 128, 4096]);  add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1021: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1018, [128, 4096]);  view_1018 = None
    permute_598: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1021, [1, 0])
    mm_216: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_598, view_534);  permute_598 = None
    permute_599: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_216, [1, 0]);  mm_216 = None
    mm_217: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1021, permute_600);  view_1021 = permute_600 = None
    view_1022: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_217, [1, 128, 4096]);  mm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_333: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1006, view_1022);  view_1006 = view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_601: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1023: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1019, [128, 4096]);  view_1019 = None
    permute_602: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1023, [1, 0])
    mm_218: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_602, view_534);  permute_602 = None
    permute_603: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_218, [1, 0]);  mm_218 = None
    mm_219: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1023, permute_604);  view_1023 = permute_604 = None
    view_1024: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_219, [1, 128, 4096]);  mm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_334: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_333, view_1024);  add_333 = view_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_605: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1025: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1020, [128, 4096]);  view_1020 = None
    permute_606: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1025, [1, 0])
    mm_220: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_606, view_534);  permute_606 = view_534 = None
    permute_607: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_220, [1, 0]);  mm_220 = None
    mm_221: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1025, permute_608);  view_1025 = permute_608 = None
    view_1026: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_221, [1, 128, 4096]);  mm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_335: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_334, view_1026);  add_334 = view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_609: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_485: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_335, primals_192);  primals_192 = None
    mul_486: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_485, 4096)
    sum_101: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
    mul_487: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_485, mul_190);  mul_485 = None
    sum_102: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
    mul_488: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, sum_102);  sum_102 = None
    sub_109: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_486, sum_101);  mul_486 = sum_101 = None
    sub_110: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_109, mul_488);  sub_109 = mul_488 = None
    mul_489: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_80, sub_110);  div_80 = sub_110 = None
    mul_490: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_335, mul_190);  mul_190 = None
    sum_103: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
    sum_104: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_336: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_324, mul_489);  add_324 = mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1027: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_336, [128, 4096])
    mm_222: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1027, permute_610);  permute_610 = None
    permute_611: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1027, [1, 0])
    mm_223: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_611, view_532);  view_532 = None
    permute_612: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_105: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True)
    view_1028: "f32[4096]" = torch.ops.aten.reshape.default(sum_105, [4096]);  sum_105 = None
    permute_613: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_1029: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_222, [1, 128, 16384]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_491: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1029, mul_186);  mul_186 = None
    mul_492: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1029, add_149);  view_1029 = add_149 = None
    mul_493: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
    sub_111: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_493);  mul_493 = None
    mul_494: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_491, sub_111);  mul_491 = sub_111 = None
    mul_495: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_494, 0.7978845608028654);  mul_494 = None
    mul_496: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_495, 0.044715)
    pow_38: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_531, 2.0);  view_531 = None
    mul_497: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
    mul_498: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_337: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_495, mul_498);  mul_495 = mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_499: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_492, 0.5);  mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_338: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_337, mul_499);  add_337 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1030: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_338, [128, 16384]);  add_338 = None
    mm_224: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1030, permute_614);  permute_614 = None
    permute_615: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1030, [1, 0])
    mm_225: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_615, view_506);  permute_615 = None
    permute_616: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_106: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1030, [0], True);  view_1030 = None
    view_1031: "f32[16384]" = torch.ops.aten.reshape.default(sum_106, [16384]);  sum_106 = None
    permute_617: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_616, [1, 0]);  permute_616 = None
    view_1032: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_224, [1, 128, 4096]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_226: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_611, view_528);  permute_611 = view_528 = None
    permute_619: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_226, [1, 0]);  mm_226 = None
    mm_227: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1027, permute_620);  view_1027 = permute_620 = None
    view_1034: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_227, [1, 128, 4096]);  mm_227 = None
    permute_621: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1035: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1034, [1, 128, 16, 256]);  view_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_622: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1035, [0, 2, 1, 3]);  view_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1036: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_622, [16, 128, 256]);  permute_622 = None
    bmm_92: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_623, view_1036);  permute_623 = None
    bmm_93: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1036, permute_624);  view_1036 = permute_624 = None
    view_1037: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_92, [1, 16, 128, 256]);  bmm_92 = None
    view_1038: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_500: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1038, alias_79);  view_1038 = None
    sum_107: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [-1], True)
    mul_501: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_79, sum_107);  alias_79 = sum_107 = None
    sub_112: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_81: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_112, primals_342);  sub_112 = primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_45: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_912, div_81, full_default_29);  slice_912 = div_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1039: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_45, [16, 128, 128]);  where_45 = None
    bmm_94: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_625, view_1039);  permute_625 = None
    bmm_95: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1039, permute_626);  view_1039 = permute_626 = None
    view_1040: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_94, [1, 16, 256, 128]);  bmm_94 = None
    view_1041: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_95, [1, 16, 128, 256]);  bmm_95 = None
    permute_627: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1040, [0, 1, 3, 2]);  view_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_628: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1041, [0, 2, 1, 3]);  view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_629: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_627, [0, 2, 1, 3]);  permute_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1381: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_628, 3, 0, 64)
    slice_1382: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_628, 3, 64, 256);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1383: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_629, 3, 0, 64)
    slice_1384: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_629, 3, 64, 256);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_502: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1381, view_515)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1042: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_502, [1, 128, 16, 32, 2]);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_36: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1042, 4, 0)
    select_37: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1042, 4, 1);  view_1042 = None
    neg_76: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_36);  select_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_288: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_292: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_339: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_288, slice_scatter_292);  slice_scatter_288 = slice_scatter_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_503: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1381, view_516);  slice_1381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_340: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_339, mul_503);  add_339 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_504: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1383, view_515);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1043: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_504, [1, 128, 16, 32, 2]);  mul_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_38: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1043, 4, 0)
    select_39: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1043, 4, 1);  view_1043 = None
    neg_77: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_38);  select_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_296: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_300: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_341: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_296, slice_scatter_300);  slice_scatter_296 = slice_scatter_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_505: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1383, view_516);  slice_1383 = view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_342: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_341, mul_505);  add_341 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_304: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1382, 3, 64, 9223372036854775807);  slice_1382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_308: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_340, 3, 0, 64);  add_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_343: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_304, slice_scatter_308);  slice_scatter_304 = slice_scatter_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_312: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1384, 3, 64, 9223372036854775807);  slice_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_316: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_342, 3, 0, 64);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_344: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_312, slice_scatter_316);  slice_scatter_312 = slice_scatter_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_630: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1037, [0, 2, 1, 3]);  view_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_236: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
    view_1044: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_236, [1, 128, 4096]);  clone_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1045: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_344, [1, 128, 4096]);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1046: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_343, [1, 128, 4096]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1047: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1044, [128, 4096]);  view_1044 = None
    permute_631: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1047, [1, 0])
    mm_228: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_631, view_506);  permute_631 = None
    permute_632: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_228, [1, 0]);  mm_228 = None
    mm_229: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1047, permute_633);  view_1047 = permute_633 = None
    view_1048: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_229, [1, 128, 4096]);  mm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_345: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1032, view_1048);  view_1032 = view_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_634: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1049: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1045, [128, 4096]);  view_1045 = None
    permute_635: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1049, [1, 0])
    mm_230: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_635, view_506);  permute_635 = None
    permute_636: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_230, [1, 0]);  mm_230 = None
    mm_231: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1049, permute_637);  view_1049 = permute_637 = None
    view_1050: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_231, [1, 128, 4096]);  mm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_346: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_345, view_1050);  add_345 = view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_638: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1051: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1046, [128, 4096]);  view_1046 = None
    permute_639: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1051, [1, 0])
    mm_232: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_639, view_506);  permute_639 = view_506 = None
    permute_640: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    mm_233: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1051, permute_641);  view_1051 = permute_641 = None
    view_1052: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_233, [1, 128, 4096]);  mm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_347: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_346, view_1052);  add_346 = view_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_642: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_507: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_347, primals_182);  primals_182 = None
    mul_508: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_507, 4096)
    sum_108: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True)
    mul_509: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_507, mul_180);  mul_507 = None
    sum_109: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
    mul_510: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, sum_109);  sum_109 = None
    sub_114: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_508, sum_108);  mul_508 = sum_108 = None
    sub_115: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_114, mul_510);  sub_114 = mul_510 = None
    mul_511: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_82, sub_115);  div_82 = sub_115 = None
    mul_512: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_347, mul_180);  mul_180 = None
    sum_110: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
    sum_111: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 1]);  add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_348: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_336, mul_511);  add_336 = mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1053: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_348, [128, 4096])
    mm_234: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1053, permute_643);  permute_643 = None
    permute_644: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1053, [1, 0])
    mm_235: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_644, view_504);  view_504 = None
    permute_645: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_112: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1053, [0], True)
    view_1054: "f32[4096]" = torch.ops.aten.reshape.default(sum_112, [4096]);  sum_112 = None
    permute_646: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_645, [1, 0]);  permute_645 = None
    view_1055: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_234, [1, 128, 16384]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_513: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1055, mul_176);  mul_176 = None
    mul_514: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1055, add_141);  view_1055 = add_141 = None
    mul_515: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
    sub_116: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_515);  mul_515 = None
    mul_516: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_513, sub_116);  mul_513 = sub_116 = None
    mul_517: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_516, 0.7978845608028654);  mul_516 = None
    mul_518: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_517, 0.044715)
    pow_39: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 2.0);  view_503 = None
    mul_519: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
    mul_520: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_349: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_517, mul_520);  mul_517 = mul_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_521: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_514, 0.5);  mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_350: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_349, mul_521);  add_349 = mul_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1056: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_350, [128, 16384]);  add_350 = None
    mm_236: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1056, permute_647);  permute_647 = None
    permute_648: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1056, [1, 0])
    mm_237: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_648, view_478);  permute_648 = None
    permute_649: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_113: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1056, [0], True);  view_1056 = None
    view_1057: "f32[16384]" = torch.ops.aten.reshape.default(sum_113, [16384]);  sum_113 = None
    permute_650: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
    view_1058: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_236, [1, 128, 4096]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_238: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_644, view_500);  permute_644 = view_500 = None
    permute_652: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_238, [1, 0]);  mm_238 = None
    mm_239: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1053, permute_653);  view_1053 = permute_653 = None
    view_1060: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_239, [1, 128, 4096]);  mm_239 = None
    permute_654: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1061: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1060, [1, 128, 16, 256]);  view_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_655: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1061, [0, 2, 1, 3]);  view_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1062: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_655, [16, 128, 256]);  permute_655 = None
    bmm_96: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_656, view_1062);  permute_656 = None
    bmm_97: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1062, permute_657);  view_1062 = permute_657 = None
    view_1063: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_96, [1, 16, 128, 256]);  bmm_96 = None
    view_1064: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_522: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1064, alias_81);  view_1064 = None
    sum_114: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [-1], True)
    mul_523: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_81, sum_114);  alias_81 = sum_114 = None
    sub_117: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_83: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_117, primals_339);  sub_117 = primals_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_46: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_864, div_83, full_default_29);  slice_864 = div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1065: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_46, [16, 128, 128]);  where_46 = None
    bmm_98: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_658, view_1065);  permute_658 = None
    bmm_99: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1065, permute_659);  view_1065 = permute_659 = None
    view_1066: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_98, [1, 16, 256, 128]);  bmm_98 = None
    view_1067: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_99, [1, 16, 128, 256]);  bmm_99 = None
    permute_660: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1066, [0, 1, 3, 2]);  view_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_661: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1067, [0, 2, 1, 3]);  view_1067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_662: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_660, [0, 2, 1, 3]);  permute_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1385: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_661, 3, 0, 64)
    slice_1386: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_661, 3, 64, 256);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1387: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_662, 3, 0, 64)
    slice_1388: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_662, 3, 64, 256);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_524: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1385, view_487)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1068: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_524, [1, 128, 16, 32, 2]);  mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_40: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1068, 4, 0)
    select_41: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1068, 4, 1);  view_1068 = None
    neg_78: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_40);  select_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_320: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_78, 3, 1, 9223372036854775807, 2);  neg_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_324: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_351: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_320, slice_scatter_324);  slice_scatter_320 = slice_scatter_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_525: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1385, view_488);  slice_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_352: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_351, mul_525);  add_351 = mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_526: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1387, view_487);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1069: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_526, [1, 128, 16, 32, 2]);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_42: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1069, 4, 0)
    select_43: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1069, 4, 1);  view_1069 = None
    neg_79: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_42);  select_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_328: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_332: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_353: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_328, slice_scatter_332);  slice_scatter_328 = slice_scatter_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_527: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1387, view_488);  slice_1387 = view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_354: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_353, mul_527);  add_353 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_336: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1386, 3, 64, 9223372036854775807);  slice_1386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_340: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_352, 3, 0, 64);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_355: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_336, slice_scatter_340);  slice_scatter_336 = slice_scatter_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_344: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1388, 3, 64, 9223372036854775807);  slice_1388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_348: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_354, 3, 0, 64);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_356: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_344, slice_scatter_348);  slice_scatter_344 = slice_scatter_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_663: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1063, [0, 2, 1, 3]);  view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_237: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
    view_1070: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_237, [1, 128, 4096]);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1071: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_356, [1, 128, 4096]);  add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1072: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_355, [1, 128, 4096]);  add_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1073: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1070, [128, 4096]);  view_1070 = None
    permute_664: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1073, [1, 0])
    mm_240: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_664, view_478);  permute_664 = None
    permute_665: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_240, [1, 0]);  mm_240 = None
    mm_241: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1073, permute_666);  view_1073 = permute_666 = None
    view_1074: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_241, [1, 128, 4096]);  mm_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_357: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1058, view_1074);  view_1058 = view_1074 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_667: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_665, [1, 0]);  permute_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1075: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1071, [128, 4096]);  view_1071 = None
    permute_668: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1075, [1, 0])
    mm_242: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_668, view_478);  permute_668 = None
    permute_669: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_242, [1, 0]);  mm_242 = None
    mm_243: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1075, permute_670);  view_1075 = permute_670 = None
    view_1076: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_358: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_357, view_1076);  add_357 = view_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_671: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1077: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1072, [128, 4096]);  view_1072 = None
    permute_672: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1077, [1, 0])
    mm_244: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_672, view_478);  permute_672 = view_478 = None
    permute_673: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    mm_245: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1077, permute_674);  view_1077 = permute_674 = None
    view_1078: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_245, [1, 128, 4096]);  mm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_359: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_358, view_1078);  add_358 = view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_675: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_529: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_359, primals_172);  primals_172 = None
    mul_530: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_529, 4096)
    sum_115: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True)
    mul_531: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_529, mul_170);  mul_529 = None
    sum_116: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True);  mul_531 = None
    mul_532: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, sum_116);  sum_116 = None
    sub_119: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_530, sum_115);  mul_530 = sum_115 = None
    sub_120: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_119, mul_532);  sub_119 = mul_532 = None
    mul_533: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_84, sub_120);  div_84 = sub_120 = None
    mul_534: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_359, mul_170);  mul_170 = None
    sum_117: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 1]);  mul_534 = None
    sum_118: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 1]);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_360: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_348, mul_533);  add_348 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1079: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_360, [128, 4096])
    mm_246: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1079, permute_676);  permute_676 = None
    permute_677: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1079, [1, 0])
    mm_247: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_677, view_476);  view_476 = None
    permute_678: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_119: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True)
    view_1080: "f32[4096]" = torch.ops.aten.reshape.default(sum_119, [4096]);  sum_119 = None
    permute_679: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_678, [1, 0]);  permute_678 = None
    view_1081: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_246, [1, 128, 16384]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_535: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1081, mul_166);  mul_166 = None
    mul_536: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1081, add_133);  view_1081 = add_133 = None
    mul_537: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
    sub_121: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_537);  mul_537 = None
    mul_538: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_535, sub_121);  mul_535 = sub_121 = None
    mul_539: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_538, 0.7978845608028654);  mul_538 = None
    mul_540: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_539, 0.044715)
    pow_40: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_475, 2.0);  view_475 = None
    mul_541: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
    mul_542: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_361: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_539, mul_542);  mul_539 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_543: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_536, 0.5);  mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_362: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_361, mul_543);  add_361 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1082: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_362, [128, 16384]);  add_362 = None
    mm_248: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1082, permute_680);  permute_680 = None
    permute_681: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1082, [1, 0])
    mm_249: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_681, view_450);  permute_681 = None
    permute_682: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_120: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1082, [0], True);  view_1082 = None
    view_1083: "f32[16384]" = torch.ops.aten.reshape.default(sum_120, [16384]);  sum_120 = None
    permute_683: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    view_1084: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_248, [1, 128, 4096]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_250: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_677, view_472);  permute_677 = view_472 = None
    permute_685: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_250, [1, 0]);  mm_250 = None
    mm_251: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1079, permute_686);  view_1079 = permute_686 = None
    view_1086: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_251, [1, 128, 4096]);  mm_251 = None
    permute_687: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_685, [1, 0]);  permute_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1087: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1086, [1, 128, 16, 256]);  view_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_688: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1087, [0, 2, 1, 3]);  view_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1088: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_688, [16, 128, 256]);  permute_688 = None
    bmm_100: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_689, view_1088);  permute_689 = None
    bmm_101: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1088, permute_690);  view_1088 = permute_690 = None
    view_1089: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_100, [1, 16, 128, 256]);  bmm_100 = None
    view_1090: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_544: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1090, alias_83);  view_1090 = None
    sum_121: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [-1], True)
    mul_545: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_83, sum_121);  alias_83 = sum_121 = None
    sub_122: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_85: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_122, primals_336);  sub_122 = primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_47: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_816, div_85, full_default_29);  slice_816 = div_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1091: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_47, [16, 128, 128]);  where_47 = None
    bmm_102: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_691, view_1091);  permute_691 = None
    bmm_103: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1091, permute_692);  view_1091 = permute_692 = None
    view_1092: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_102, [1, 16, 256, 128]);  bmm_102 = None
    view_1093: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_103, [1, 16, 128, 256]);  bmm_103 = None
    permute_693: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1092, [0, 1, 3, 2]);  view_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_694: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1093, [0, 2, 1, 3]);  view_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_695: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_693, [0, 2, 1, 3]);  permute_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1389: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_694, 3, 0, 64)
    slice_1390: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_694, 3, 64, 256);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1391: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_695, 3, 0, 64)
    slice_1392: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_695, 3, 64, 256);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_546: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1389, view_459)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1094: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_546, [1, 128, 16, 32, 2]);  mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_44: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1094, 4, 0)
    select_45: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1094, 4, 1);  view_1094 = None
    neg_80: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_44);  select_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_352: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_356: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_363: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_352, slice_scatter_356);  slice_scatter_352 = slice_scatter_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_547: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1389, view_460);  slice_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_364: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_363, mul_547);  add_363 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_548: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1391, view_459);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1095: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_548, [1, 128, 16, 32, 2]);  mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_46: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1095, 4, 0)
    select_47: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1095, 4, 1);  view_1095 = None
    neg_81: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_46);  select_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_360: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_81, 3, 1, 9223372036854775807, 2);  neg_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_364: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_365: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_360, slice_scatter_364);  slice_scatter_360 = slice_scatter_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_549: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1391, view_460);  slice_1391 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_366: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_365, mul_549);  add_365 = mul_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_368: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1390, 3, 64, 9223372036854775807);  slice_1390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_372: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_364, 3, 0, 64);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_367: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_368, slice_scatter_372);  slice_scatter_368 = slice_scatter_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_376: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1392, 3, 64, 9223372036854775807);  slice_1392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_380: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_366, 3, 0, 64);  add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_368: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_376, slice_scatter_380);  slice_scatter_376 = slice_scatter_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_696: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1089, [0, 2, 1, 3]);  view_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_238: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
    view_1096: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_238, [1, 128, 4096]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1097: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_368, [1, 128, 4096]);  add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1098: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_367, [1, 128, 4096]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1099: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1096, [128, 4096]);  view_1096 = None
    permute_697: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_252: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_697, view_450);  permute_697 = None
    permute_698: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_252, [1, 0]);  mm_252 = None
    mm_253: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1099, permute_699);  view_1099 = permute_699 = None
    view_1100: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_253, [1, 128, 4096]);  mm_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_369: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1084, view_1100);  view_1084 = view_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_700: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1101: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1097, [128, 4096]);  view_1097 = None
    permute_701: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1101, [1, 0])
    mm_254: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_701, view_450);  permute_701 = None
    permute_702: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_254, [1, 0]);  mm_254 = None
    mm_255: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1101, permute_703);  view_1101 = permute_703 = None
    view_1102: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_370: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_369, view_1102);  add_369 = view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_704: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1103: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1098, [128, 4096]);  view_1098 = None
    permute_705: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1103, [1, 0])
    mm_256: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_705, view_450);  permute_705 = view_450 = None
    permute_706: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    mm_257: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1103, permute_707);  view_1103 = permute_707 = None
    view_1104: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_257, [1, 128, 4096]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_371: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_370, view_1104);  add_370 = view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_708: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_551: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_371, primals_162);  primals_162 = None
    mul_552: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_551, 4096)
    sum_122: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
    mul_553: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_551, mul_160);  mul_551 = None
    sum_123: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
    mul_554: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, sum_123);  sum_123 = None
    sub_124: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_552, sum_122);  mul_552 = sum_122 = None
    sub_125: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_124, mul_554);  sub_124 = mul_554 = None
    mul_555: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_86, sub_125);  div_86 = sub_125 = None
    mul_556: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_371, mul_160);  mul_160 = None
    sum_124: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
    sum_125: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_371, [0, 1]);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_372: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_360, mul_555);  add_360 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1105: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_372, [128, 4096])
    mm_258: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1105, permute_709);  permute_709 = None
    permute_710: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1105, [1, 0])
    mm_259: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_710, view_448);  view_448 = None
    permute_711: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1105, [0], True)
    view_1106: "f32[4096]" = torch.ops.aten.reshape.default(sum_126, [4096]);  sum_126 = None
    permute_712: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_711, [1, 0]);  permute_711 = None
    view_1107: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_258, [1, 128, 16384]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_557: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1107, mul_156);  mul_156 = None
    mul_558: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1107, add_125);  view_1107 = add_125 = None
    mul_559: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
    sub_126: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_559);  mul_559 = None
    mul_560: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_557, sub_126);  mul_557 = sub_126 = None
    mul_561: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_560, 0.7978845608028654);  mul_560 = None
    mul_562: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_561, 0.044715)
    pow_41: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_447, 2.0);  view_447 = None
    mul_563: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
    mul_564: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_562, mul_563);  mul_562 = mul_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_373: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_561, mul_564);  mul_561 = mul_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_565: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_558, 0.5);  mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_374: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_373, mul_565);  add_373 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1108: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_374, [128, 16384]);  add_374 = None
    mm_260: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1108, permute_713);  permute_713 = None
    permute_714: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1108, [1, 0])
    mm_261: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_714, view_422);  permute_714 = None
    permute_715: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_127: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1108, [0], True);  view_1108 = None
    view_1109: "f32[16384]" = torch.ops.aten.reshape.default(sum_127, [16384]);  sum_127 = None
    permute_716: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None
    view_1110: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_260, [1, 128, 4096]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_262: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_710, view_444);  permute_710 = view_444 = None
    permute_718: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_262, [1, 0]);  mm_262 = None
    mm_263: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1105, permute_719);  view_1105 = permute_719 = None
    view_1112: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_263, [1, 128, 4096]);  mm_263 = None
    permute_720: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1113: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1112, [1, 128, 16, 256]);  view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_721: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1113, [0, 2, 1, 3]);  view_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1114: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_721, [16, 128, 256]);  permute_721 = None
    bmm_104: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_722, view_1114);  permute_722 = None
    bmm_105: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1114, permute_723);  view_1114 = permute_723 = None
    view_1115: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_104, [1, 16, 128, 256]);  bmm_104 = None
    view_1116: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_566: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1116, alias_85);  view_1116 = None
    sum_128: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [-1], True)
    mul_567: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_85, sum_128);  alias_85 = sum_128 = None
    sub_127: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_87: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_127, primals_333);  sub_127 = primals_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_48: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_768, div_87, full_default_29);  slice_768 = div_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1117: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_48, [16, 128, 128]);  where_48 = None
    bmm_106: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_724, view_1117);  permute_724 = None
    bmm_107: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1117, permute_725);  view_1117 = permute_725 = None
    view_1118: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_106, [1, 16, 256, 128]);  bmm_106 = None
    view_1119: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_107, [1, 16, 128, 256]);  bmm_107 = None
    permute_726: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1118, [0, 1, 3, 2]);  view_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_727: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1119, [0, 2, 1, 3]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_728: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_726, [0, 2, 1, 3]);  permute_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1393: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_727, 3, 0, 64)
    slice_1394: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_727, 3, 64, 256);  permute_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1395: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_728, 3, 0, 64)
    slice_1396: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_728, 3, 64, 256);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_568: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1393, view_431)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1120: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_568, [1, 128, 16, 32, 2]);  mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_48: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1120, 4, 0)
    select_49: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1120, 4, 1);  view_1120 = None
    neg_82: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_48);  select_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_384: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_388: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_375: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_384, slice_scatter_388);  slice_scatter_384 = slice_scatter_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_569: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1393, view_432);  slice_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_376: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_375, mul_569);  add_375 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_570: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1395, view_431);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1121: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_570, [1, 128, 16, 32, 2]);  mul_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_50: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1121, 4, 0)
    select_51: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1121, 4, 1);  view_1121 = None
    neg_83: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_50);  select_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_392: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_396: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_377: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_392, slice_scatter_396);  slice_scatter_392 = slice_scatter_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_571: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1395, view_432);  slice_1395 = view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_378: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_377, mul_571);  add_377 = mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_400: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1394, 3, 64, 9223372036854775807);  slice_1394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_404: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_376, 3, 0, 64);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_379: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_400, slice_scatter_404);  slice_scatter_400 = slice_scatter_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_408: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1396, 3, 64, 9223372036854775807);  slice_1396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_412: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_378, 3, 0, 64);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_380: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_408, slice_scatter_412);  slice_scatter_408 = slice_scatter_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_729: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1115, [0, 2, 1, 3]);  view_1115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_239: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
    view_1122: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_239, [1, 128, 4096]);  clone_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1123: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_380, [1, 128, 4096]);  add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1124: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_379, [1, 128, 4096]);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1125: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1122, [128, 4096]);  view_1122 = None
    permute_730: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1125, [1, 0])
    mm_264: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_730, view_422);  permute_730 = None
    permute_731: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_264, [1, 0]);  mm_264 = None
    mm_265: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1125, permute_732);  view_1125 = permute_732 = None
    view_1126: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_265, [1, 128, 4096]);  mm_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_381: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1110, view_1126);  view_1110 = view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_733: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1127: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1123, [128, 4096]);  view_1123 = None
    permute_734: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1127, [1, 0])
    mm_266: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_734, view_422);  permute_734 = None
    permute_735: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_266, [1, 0]);  mm_266 = None
    mm_267: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1127, permute_736);  view_1127 = permute_736 = None
    view_1128: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_382: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_381, view_1128);  add_381 = view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_737: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_735, [1, 0]);  permute_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1129: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1124, [128, 4096]);  view_1124 = None
    permute_738: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1129, [1, 0])
    mm_268: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_738, view_422);  permute_738 = view_422 = None
    permute_739: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    mm_269: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1129, permute_740);  view_1129 = permute_740 = None
    view_1130: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_269, [1, 128, 4096]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_383: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_382, view_1130);  add_382 = view_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_741: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_739, [1, 0]);  permute_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_573: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_383, primals_152);  primals_152 = None
    mul_574: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_573, 4096)
    sum_129: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True)
    mul_575: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_573, mul_150);  mul_573 = None
    sum_130: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_575, [2], True);  mul_575 = None
    mul_576: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, sum_130);  sum_130 = None
    sub_129: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_574, sum_129);  mul_574 = sum_129 = None
    sub_130: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_129, mul_576);  sub_129 = mul_576 = None
    mul_577: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_88, sub_130);  div_88 = sub_130 = None
    mul_578: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_383, mul_150);  mul_150 = None
    sum_131: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1]);  mul_578 = None
    sum_132: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_383, [0, 1]);  add_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_384: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_372, mul_577);  add_372 = mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1131: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_384, [128, 4096])
    mm_270: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1131, permute_742);  permute_742 = None
    permute_743: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1131, [1, 0])
    mm_271: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_743, view_420);  view_420 = None
    permute_744: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_133: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True)
    view_1132: "f32[4096]" = torch.ops.aten.reshape.default(sum_133, [4096]);  sum_133 = None
    permute_745: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_744, [1, 0]);  permute_744 = None
    view_1133: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_270, [1, 128, 16384]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_579: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1133, mul_146);  mul_146 = None
    mul_580: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1133, add_117);  view_1133 = add_117 = None
    mul_581: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
    sub_131: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_581);  mul_581 = None
    mul_582: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_579, sub_131);  mul_579 = sub_131 = None
    mul_583: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_582, 0.7978845608028654);  mul_582 = None
    mul_584: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_583, 0.044715)
    pow_42: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_419, 2.0);  view_419 = None
    mul_585: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
    mul_586: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_385: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_583, mul_586);  mul_583 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_587: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_580, 0.5);  mul_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_386: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_385, mul_587);  add_385 = mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1134: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_386, [128, 16384]);  add_386 = None
    mm_272: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1134, permute_746);  permute_746 = None
    permute_747: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_273: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_747, view_394);  permute_747 = None
    permute_748: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_134: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1134, [0], True);  view_1134 = None
    view_1135: "f32[16384]" = torch.ops.aten.reshape.default(sum_134, [16384]);  sum_134 = None
    permute_749: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_1136: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_272, [1, 128, 4096]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_274: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_743, view_416);  permute_743 = view_416 = None
    permute_751: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_274, [1, 0]);  mm_274 = None
    mm_275: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1131, permute_752);  view_1131 = permute_752 = None
    view_1138: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_275, [1, 128, 4096]);  mm_275 = None
    permute_753: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_751, [1, 0]);  permute_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1139: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1138, [1, 128, 16, 256]);  view_1138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_754: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1, 3]);  view_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1140: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_754, [16, 128, 256]);  permute_754 = None
    bmm_108: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_755, view_1140);  permute_755 = None
    bmm_109: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1140, permute_756);  view_1140 = permute_756 = None
    view_1141: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_108, [1, 16, 128, 256]);  bmm_108 = None
    view_1142: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_588: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1142, alias_87);  view_1142 = None
    sum_135: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [-1], True)
    mul_589: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_87, sum_135);  alias_87 = sum_135 = None
    sub_132: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_89: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_132, primals_330);  sub_132 = primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_49: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_720, div_89, full_default_29);  slice_720 = div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1143: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_49, [16, 128, 128]);  where_49 = None
    bmm_110: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_757, view_1143);  permute_757 = None
    bmm_111: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1143, permute_758);  view_1143 = permute_758 = None
    view_1144: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_110, [1, 16, 256, 128]);  bmm_110 = None
    view_1145: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_111, [1, 16, 128, 256]);  bmm_111 = None
    permute_759: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1144, [0, 1, 3, 2]);  view_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_760: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1145, [0, 2, 1, 3]);  view_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_761: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_759, [0, 2, 1, 3]);  permute_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1397: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_760, 3, 0, 64)
    slice_1398: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_760, 3, 64, 256);  permute_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1399: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_761, 3, 0, 64)
    slice_1400: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_761, 3, 64, 256);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_590: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1397, view_403)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1146: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_590, [1, 128, 16, 32, 2]);  mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_52: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1146, 4, 0)
    select_53: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1146, 4, 1);  view_1146 = None
    neg_84: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_52);  select_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_416: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_84, 3, 1, 9223372036854775807, 2);  neg_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_420: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_387: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_416, slice_scatter_420);  slice_scatter_416 = slice_scatter_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_591: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1397, view_404);  slice_1397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_388: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_387, mul_591);  add_387 = mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_592: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1399, view_403);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1147: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_592, [1, 128, 16, 32, 2]);  mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_54: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1147, 4, 0)
    select_55: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1147, 4, 1);  view_1147 = None
    neg_85: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_54);  select_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_424: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_85, 3, 1, 9223372036854775807, 2);  neg_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_428: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_389: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_424, slice_scatter_428);  slice_scatter_424 = slice_scatter_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_593: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1399, view_404);  slice_1399 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_390: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_389, mul_593);  add_389 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_432: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1398, 3, 64, 9223372036854775807);  slice_1398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_436: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_388, 3, 0, 64);  add_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_391: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_432, slice_scatter_436);  slice_scatter_432 = slice_scatter_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_440: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1400, 3, 64, 9223372036854775807);  slice_1400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_444: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_390, 3, 0, 64);  add_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_392: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_440, slice_scatter_444);  slice_scatter_440 = slice_scatter_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_762: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1141, [0, 2, 1, 3]);  view_1141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_240: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
    view_1148: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_240, [1, 128, 4096]);  clone_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1149: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_392, [1, 128, 4096]);  add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1150: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_391, [1, 128, 4096]);  add_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1151: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1148, [128, 4096]);  view_1148 = None
    permute_763: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1151, [1, 0])
    mm_276: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_763, view_394);  permute_763 = None
    permute_764: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_276, [1, 0]);  mm_276 = None
    mm_277: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1151, permute_765);  view_1151 = permute_765 = None
    view_1152: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_277, [1, 128, 4096]);  mm_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_393: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1136, view_1152);  view_1136 = view_1152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_766: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_764, [1, 0]);  permute_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1153: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1149, [128, 4096]);  view_1149 = None
    permute_767: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1153, [1, 0])
    mm_278: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_767, view_394);  permute_767 = None
    permute_768: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_278, [1, 0]);  mm_278 = None
    mm_279: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1153, permute_769);  view_1153 = permute_769 = None
    view_1154: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_394: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_393, view_1154);  add_393 = view_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_770: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_768, [1, 0]);  permute_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1155: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1150, [128, 4096]);  view_1150 = None
    permute_771: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1155, [1, 0])
    mm_280: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_771, view_394);  permute_771 = view_394 = None
    permute_772: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_280, [1, 0]);  mm_280 = None
    mm_281: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1155, permute_773);  view_1155 = permute_773 = None
    view_1156: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_281, [1, 128, 4096]);  mm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_395: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_394, view_1156);  add_394 = view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_774: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_772, [1, 0]);  permute_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_595: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_395, primals_142);  primals_142 = None
    mul_596: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_595, 4096)
    sum_136: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True)
    mul_597: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_595, mul_140);  mul_595 = None
    sum_137: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True);  mul_597 = None
    mul_598: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, sum_137);  sum_137 = None
    sub_134: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_596, sum_136);  mul_596 = sum_136 = None
    sub_135: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_134, mul_598);  sub_134 = mul_598 = None
    mul_599: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_90, sub_135);  div_90 = sub_135 = None
    mul_600: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_395, mul_140);  mul_140 = None
    sum_138: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 1]);  mul_600 = None
    sum_139: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_395, [0, 1]);  add_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_396: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_384, mul_599);  add_384 = mul_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1157: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_396, [128, 4096])
    mm_282: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1157, permute_775);  permute_775 = None
    permute_776: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1157, [1, 0])
    mm_283: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_776, view_392);  view_392 = None
    permute_777: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True)
    view_1158: "f32[4096]" = torch.ops.aten.reshape.default(sum_140, [4096]);  sum_140 = None
    permute_778: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_1159: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_282, [1, 128, 16384]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_601: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1159, mul_136);  mul_136 = None
    mul_602: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1159, add_109);  view_1159 = add_109 = None
    mul_603: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
    sub_136: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_603);  mul_603 = None
    mul_604: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_601, sub_136);  mul_601 = sub_136 = None
    mul_605: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_604, 0.7978845608028654);  mul_604 = None
    mul_606: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_605, 0.044715)
    pow_43: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_391, 2.0);  view_391 = None
    mul_607: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
    mul_608: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_397: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_605, mul_608);  mul_605 = mul_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_609: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_602, 0.5);  mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_398: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_397, mul_609);  add_397 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1160: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_398, [128, 16384]);  add_398 = None
    mm_284: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1160, permute_779);  permute_779 = None
    permute_780: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1160, [1, 0])
    mm_285: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_780, view_366);  permute_780 = None
    permute_781: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_141: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
    view_1161: "f32[16384]" = torch.ops.aten.reshape.default(sum_141, [16384]);  sum_141 = None
    permute_782: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_1162: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_284, [1, 128, 4096]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_286: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_776, view_388);  permute_776 = view_388 = None
    permute_784: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_286, [1, 0]);  mm_286 = None
    mm_287: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1157, permute_785);  view_1157 = permute_785 = None
    view_1164: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_287, [1, 128, 4096]);  mm_287 = None
    permute_786: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_784, [1, 0]);  permute_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1165: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1164, [1, 128, 16, 256]);  view_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_787: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1165, [0, 2, 1, 3]);  view_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1166: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_787, [16, 128, 256]);  permute_787 = None
    bmm_112: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_788, view_1166);  permute_788 = None
    bmm_113: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1166, permute_789);  view_1166 = permute_789 = None
    view_1167: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_112, [1, 16, 128, 256]);  bmm_112 = None
    view_1168: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_610: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1168, alias_89);  view_1168 = None
    sum_142: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [-1], True)
    mul_611: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_142);  alias_89 = sum_142 = None
    sub_137: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_91: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_137, primals_327);  sub_137 = primals_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_50: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_672, div_91, full_default_29);  slice_672 = div_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1169: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_50, [16, 128, 128]);  where_50 = None
    bmm_114: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_790, view_1169);  permute_790 = None
    bmm_115: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1169, permute_791);  view_1169 = permute_791 = None
    view_1170: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_114, [1, 16, 256, 128]);  bmm_114 = None
    view_1171: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_115, [1, 16, 128, 256]);  bmm_115 = None
    permute_792: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1170, [0, 1, 3, 2]);  view_1170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_793: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1171, [0, 2, 1, 3]);  view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_794: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_792, [0, 2, 1, 3]);  permute_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1401: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_793, 3, 0, 64)
    slice_1402: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_793, 3, 64, 256);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1403: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_794, 3, 0, 64)
    slice_1404: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_794, 3, 64, 256);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_612: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1401, view_375)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1172: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_612, [1, 128, 16, 32, 2]);  mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_56: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1172, 4, 0)
    select_57: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1172, 4, 1);  view_1172 = None
    neg_86: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_56);  select_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_448: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_86, 3, 1, 9223372036854775807, 2);  neg_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_452: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_399: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_448, slice_scatter_452);  slice_scatter_448 = slice_scatter_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_613: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1401, view_376);  slice_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_400: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_399, mul_613);  add_399 = mul_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_614: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1403, view_375);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1173: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_614, [1, 128, 16, 32, 2]);  mul_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_58: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1173, 4, 0)
    select_59: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1173, 4, 1);  view_1173 = None
    neg_87: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_58);  select_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_456: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_87, 3, 1, 9223372036854775807, 2);  neg_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_460: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_401: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_456, slice_scatter_460);  slice_scatter_456 = slice_scatter_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_615: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1403, view_376);  slice_1403 = view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_402: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_401, mul_615);  add_401 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_464: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1402, 3, 64, 9223372036854775807);  slice_1402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_468: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_400, 3, 0, 64);  add_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_403: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_464, slice_scatter_468);  slice_scatter_464 = slice_scatter_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_472: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1404, 3, 64, 9223372036854775807);  slice_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_476: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_402, 3, 0, 64);  add_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_404: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_472, slice_scatter_476);  slice_scatter_472 = slice_scatter_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_795: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1167, [0, 2, 1, 3]);  view_1167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_241: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
    view_1174: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_241, [1, 128, 4096]);  clone_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1175: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_404, [1, 128, 4096]);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1176: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_403, [1, 128, 4096]);  add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1177: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1174, [128, 4096]);  view_1174 = None
    permute_796: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1177, [1, 0])
    mm_288: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_796, view_366);  permute_796 = None
    permute_797: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_288, [1, 0]);  mm_288 = None
    mm_289: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1177, permute_798);  view_1177 = permute_798 = None
    view_1178: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_289, [1, 128, 4096]);  mm_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_405: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1162, view_1178);  view_1162 = view_1178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_799: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1179: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1175, [128, 4096]);  view_1175 = None
    permute_800: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1179, [1, 0])
    mm_290: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_800, view_366);  permute_800 = None
    permute_801: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_290, [1, 0]);  mm_290 = None
    mm_291: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1179, permute_802);  view_1179 = permute_802 = None
    view_1180: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_406: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_405, view_1180);  add_405 = view_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_803: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_801, [1, 0]);  permute_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1181: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1176, [128, 4096]);  view_1176 = None
    permute_804: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1181, [1, 0])
    mm_292: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_804, view_366);  permute_804 = view_366 = None
    permute_805: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_292, [1, 0]);  mm_292 = None
    mm_293: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1181, permute_806);  view_1181 = permute_806 = None
    view_1182: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_293, [1, 128, 4096]);  mm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_407: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_406, view_1182);  add_406 = view_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_807: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_805, [1, 0]);  permute_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_617: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_407, primals_132);  primals_132 = None
    mul_618: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_617, 4096)
    sum_143: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True)
    mul_619: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_617, mul_130);  mul_617 = None
    sum_144: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [2], True);  mul_619 = None
    mul_620: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, sum_144);  sum_144 = None
    sub_139: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_618, sum_143);  mul_618 = sum_143 = None
    sub_140: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_139, mul_620);  sub_139 = mul_620 = None
    mul_621: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_92, sub_140);  div_92 = sub_140 = None
    mul_622: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_407, mul_130);  mul_130 = None
    sum_145: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 1]);  mul_622 = None
    sum_146: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_407, [0, 1]);  add_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_408: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_396, mul_621);  add_396 = mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1183: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_408, [128, 4096])
    mm_294: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1183, permute_808);  permute_808 = None
    permute_809: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1183, [1, 0])
    mm_295: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_809, view_364);  view_364 = None
    permute_810: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
    sum_147: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True)
    view_1184: "f32[4096]" = torch.ops.aten.reshape.default(sum_147, [4096]);  sum_147 = None
    permute_811: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
    view_1185: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_294, [1, 128, 16384]);  mm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_623: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1185, mul_126);  mul_126 = None
    mul_624: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1185, add_101);  view_1185 = add_101 = None
    mul_625: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
    sub_141: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_625);  mul_625 = None
    mul_626: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_623, sub_141);  mul_623 = sub_141 = None
    mul_627: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_626, 0.7978845608028654);  mul_626 = None
    mul_628: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_627, 0.044715)
    pow_44: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_363, 2.0);  view_363 = None
    mul_629: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
    mul_630: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_409: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_627, mul_630);  mul_627 = mul_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_631: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_624, 0.5);  mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_410: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_409, mul_631);  add_409 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1186: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_410, [128, 16384]);  add_410 = None
    mm_296: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1186, permute_812);  permute_812 = None
    permute_813: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1186, [1, 0])
    mm_297: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_813, view_338);  permute_813 = None
    permute_814: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    sum_148: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1186, [0], True);  view_1186 = None
    view_1187: "f32[16384]" = torch.ops.aten.reshape.default(sum_148, [16384]);  sum_148 = None
    permute_815: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_1188: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_296, [1, 128, 4096]);  mm_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_298: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_809, view_360);  permute_809 = view_360 = None
    permute_817: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_298, [1, 0]);  mm_298 = None
    mm_299: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1183, permute_818);  view_1183 = permute_818 = None
    view_1190: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_299, [1, 128, 4096]);  mm_299 = None
    permute_819: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_817, [1, 0]);  permute_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1191: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1190, [1, 128, 16, 256]);  view_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_820: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1191, [0, 2, 1, 3]);  view_1191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1192: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_820, [16, 128, 256]);  permute_820 = None
    bmm_116: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_821, view_1192);  permute_821 = None
    bmm_117: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1192, permute_822);  view_1192 = permute_822 = None
    view_1193: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_116, [1, 16, 128, 256]);  bmm_116 = None
    view_1194: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_632: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1194, alias_91);  view_1194 = None
    sum_149: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [-1], True)
    mul_633: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_91, sum_149);  alias_91 = sum_149 = None
    sub_142: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_93: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_142, primals_324);  sub_142 = primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_51: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_624, div_93, full_default_29);  slice_624 = div_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1195: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_51, [16, 128, 128]);  where_51 = None
    bmm_118: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_823, view_1195);  permute_823 = None
    bmm_119: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1195, permute_824);  view_1195 = permute_824 = None
    view_1196: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_118, [1, 16, 256, 128]);  bmm_118 = None
    view_1197: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_119, [1, 16, 128, 256]);  bmm_119 = None
    permute_825: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1196, [0, 1, 3, 2]);  view_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_826: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1197, [0, 2, 1, 3]);  view_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_827: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_825, [0, 2, 1, 3]);  permute_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1405: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_826, 3, 0, 64)
    slice_1406: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_826, 3, 64, 256);  permute_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1407: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_827, 3, 0, 64)
    slice_1408: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_827, 3, 64, 256);  permute_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_634: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1405, view_347)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1198: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_634, [1, 128, 16, 32, 2]);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_60: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1198, 4, 0)
    select_61: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1198, 4, 1);  view_1198 = None
    neg_88: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_60);  select_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_480: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_88, 3, 1, 9223372036854775807, 2);  neg_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_484: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_411: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_480, slice_scatter_484);  slice_scatter_480 = slice_scatter_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_635: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1405, view_348);  slice_1405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_412: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_411, mul_635);  add_411 = mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_636: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1407, view_347);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1199: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_636, [1, 128, 16, 32, 2]);  mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_62: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1199, 4, 0)
    select_63: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1199, 4, 1);  view_1199 = None
    neg_89: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_62);  select_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_488: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_89, 3, 1, 9223372036854775807, 2);  neg_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_492: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_413: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_488, slice_scatter_492);  slice_scatter_488 = slice_scatter_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_637: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1407, view_348);  slice_1407 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_414: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_413, mul_637);  add_413 = mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_496: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1406, 3, 64, 9223372036854775807);  slice_1406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_500: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_412, 3, 0, 64);  add_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_415: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_496, slice_scatter_500);  slice_scatter_496 = slice_scatter_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_504: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1408, 3, 64, 9223372036854775807);  slice_1408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_508: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_414, 3, 0, 64);  add_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_416: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_504, slice_scatter_508);  slice_scatter_504 = slice_scatter_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_828: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1193, [0, 2, 1, 3]);  view_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_242: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
    view_1200: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_242, [1, 128, 4096]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1201: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_416, [1, 128, 4096]);  add_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1202: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_415, [1, 128, 4096]);  add_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1203: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1200, [128, 4096]);  view_1200 = None
    permute_829: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1203, [1, 0])
    mm_300: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_829, view_338);  permute_829 = None
    permute_830: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_300, [1, 0]);  mm_300 = None
    mm_301: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1203, permute_831);  view_1203 = permute_831 = None
    view_1204: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_301, [1, 128, 4096]);  mm_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_417: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1188, view_1204);  view_1188 = view_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_832: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_830, [1, 0]);  permute_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1205: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1201, [128, 4096]);  view_1201 = None
    permute_833: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1205, [1, 0])
    mm_302: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_833, view_338);  permute_833 = None
    permute_834: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_302, [1, 0]);  mm_302 = None
    mm_303: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1205, permute_835);  view_1205 = permute_835 = None
    view_1206: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_418: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_417, view_1206);  add_417 = view_1206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_836: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1207: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1202, [128, 4096]);  view_1202 = None
    permute_837: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1207, [1, 0])
    mm_304: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_837, view_338);  permute_837 = view_338 = None
    permute_838: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_304, [1, 0]);  mm_304 = None
    mm_305: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1207, permute_839);  view_1207 = permute_839 = None
    view_1208: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_305, [1, 128, 4096]);  mm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_419: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_418, view_1208);  add_418 = view_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_840: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_838, [1, 0]);  permute_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_639: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_419, primals_122);  primals_122 = None
    mul_640: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_639, 4096)
    sum_150: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_639, [2], True)
    mul_641: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_639, mul_120);  mul_639 = None
    sum_151: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_641, [2], True);  mul_641 = None
    mul_642: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, sum_151);  sum_151 = None
    sub_144: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_640, sum_150);  mul_640 = sum_150 = None
    sub_145: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_144, mul_642);  sub_144 = mul_642 = None
    mul_643: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_94, sub_145);  div_94 = sub_145 = None
    mul_644: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_419, mul_120);  mul_120 = None
    sum_152: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 1]);  mul_644 = None
    sum_153: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_419, [0, 1]);  add_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_420: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_408, mul_643);  add_408 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1209: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_420, [128, 4096])
    mm_306: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1209, permute_841);  permute_841 = None
    permute_842: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1209, [1, 0])
    mm_307: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_842, view_336);  view_336 = None
    permute_843: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
    sum_154: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1209, [0], True)
    view_1210: "f32[4096]" = torch.ops.aten.reshape.default(sum_154, [4096]);  sum_154 = None
    permute_844: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    view_1211: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_306, [1, 128, 16384]);  mm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_645: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1211, mul_116);  mul_116 = None
    mul_646: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1211, add_93);  view_1211 = add_93 = None
    mul_647: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
    sub_146: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_647);  mul_647 = None
    mul_648: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_645, sub_146);  mul_645 = sub_146 = None
    mul_649: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_648, 0.7978845608028654);  mul_648 = None
    mul_650: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_649, 0.044715)
    pow_45: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_335, 2.0);  view_335 = None
    mul_651: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
    mul_652: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_650, mul_651);  mul_650 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_421: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_649, mul_652);  mul_649 = mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_653: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_646, 0.5);  mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_422: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_421, mul_653);  add_421 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1212: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_422, [128, 16384]);  add_422 = None
    mm_308: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1212, permute_845);  permute_845 = None
    permute_846: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1212, [1, 0])
    mm_309: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_846, view_310);  permute_846 = None
    permute_847: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    sum_155: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1212, [0], True);  view_1212 = None
    view_1213: "f32[16384]" = torch.ops.aten.reshape.default(sum_155, [16384]);  sum_155 = None
    permute_848: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    view_1214: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_308, [1, 128, 4096]);  mm_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_310: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_842, view_332);  permute_842 = view_332 = None
    permute_850: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_310, [1, 0]);  mm_310 = None
    mm_311: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1209, permute_851);  view_1209 = permute_851 = None
    view_1216: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_311, [1, 128, 4096]);  mm_311 = None
    permute_852: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_850, [1, 0]);  permute_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1217: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1216, [1, 128, 16, 256]);  view_1216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_853: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1217, [0, 2, 1, 3]);  view_1217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1218: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_853, [16, 128, 256]);  permute_853 = None
    bmm_120: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_854, view_1218);  permute_854 = None
    bmm_121: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1218, permute_855);  view_1218 = permute_855 = None
    view_1219: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_120, [1, 16, 128, 256]);  bmm_120 = None
    view_1220: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_654: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1220, alias_93);  view_1220 = None
    sum_156: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_654, [-1], True)
    mul_655: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_93, sum_156);  alias_93 = sum_156 = None
    sub_147: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_95: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_147, primals_321);  sub_147 = primals_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_52: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_576, div_95, full_default_29);  slice_576 = div_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1221: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_52, [16, 128, 128]);  where_52 = None
    bmm_122: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_856, view_1221);  permute_856 = None
    bmm_123: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1221, permute_857);  view_1221 = permute_857 = None
    view_1222: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_122, [1, 16, 256, 128]);  bmm_122 = None
    view_1223: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_123, [1, 16, 128, 256]);  bmm_123 = None
    permute_858: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1222, [0, 1, 3, 2]);  view_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_859: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1223, [0, 2, 1, 3]);  view_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_860: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_858, [0, 2, 1, 3]);  permute_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1409: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_859, 3, 0, 64)
    slice_1410: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_859, 3, 64, 256);  permute_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1411: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_860, 3, 0, 64)
    slice_1412: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_860, 3, 64, 256);  permute_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_656: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1409, view_319)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1224: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_656, [1, 128, 16, 32, 2]);  mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_64: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1224, 4, 0)
    select_65: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1224, 4, 1);  view_1224 = None
    neg_90: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_64);  select_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_512: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_90, 3, 1, 9223372036854775807, 2);  neg_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_516: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_423: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_512, slice_scatter_516);  slice_scatter_512 = slice_scatter_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_657: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1409, view_320);  slice_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_424: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_423, mul_657);  add_423 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_658: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1411, view_319);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1225: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_658, [1, 128, 16, 32, 2]);  mul_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_66: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1225, 4, 0)
    select_67: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1225, 4, 1);  view_1225 = None
    neg_91: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_66);  select_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_520: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_91, 3, 1, 9223372036854775807, 2);  neg_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_524: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_425: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_520, slice_scatter_524);  slice_scatter_520 = slice_scatter_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_659: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1411, view_320);  slice_1411 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_426: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_425, mul_659);  add_425 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_528: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1410, 3, 64, 9223372036854775807);  slice_1410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_532: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_424, 3, 0, 64);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_427: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_528, slice_scatter_532);  slice_scatter_528 = slice_scatter_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_536: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1412, 3, 64, 9223372036854775807);  slice_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_540: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_426, 3, 0, 64);  add_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_428: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_536, slice_scatter_540);  slice_scatter_536 = slice_scatter_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_861: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1219, [0, 2, 1, 3]);  view_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_243: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
    view_1226: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_243, [1, 128, 4096]);  clone_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1227: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_428, [1, 128, 4096]);  add_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1228: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_427, [1, 128, 4096]);  add_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1229: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1226, [128, 4096]);  view_1226 = None
    permute_862: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1229, [1, 0])
    mm_312: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_862, view_310);  permute_862 = None
    permute_863: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_312, [1, 0]);  mm_312 = None
    mm_313: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1229, permute_864);  view_1229 = permute_864 = None
    view_1230: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_313, [1, 128, 4096]);  mm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_429: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1214, view_1230);  view_1214 = view_1230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_865: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_863, [1, 0]);  permute_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1231: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1227, [128, 4096]);  view_1227 = None
    permute_866: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1231, [1, 0])
    mm_314: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_866, view_310);  permute_866 = None
    permute_867: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_314, [1, 0]);  mm_314 = None
    mm_315: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1231, permute_868);  view_1231 = permute_868 = None
    view_1232: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_430: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_429, view_1232);  add_429 = view_1232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_869: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1233: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1228, [128, 4096]);  view_1228 = None
    permute_870: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1233, [1, 0])
    mm_316: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_870, view_310);  permute_870 = view_310 = None
    permute_871: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_316, [1, 0]);  mm_316 = None
    mm_317: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1233, permute_872);  view_1233 = permute_872 = None
    view_1234: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_317, [1, 128, 4096]);  mm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_431: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_430, view_1234);  add_430 = view_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_873: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_661: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_431, primals_112);  primals_112 = None
    mul_662: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_661, 4096)
    sum_157: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True)
    mul_663: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_661, mul_110);  mul_661 = None
    sum_158: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True);  mul_663 = None
    mul_664: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, sum_158);  sum_158 = None
    sub_149: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_662, sum_157);  mul_662 = sum_157 = None
    sub_150: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_149, mul_664);  sub_149 = mul_664 = None
    mul_665: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_96, sub_150);  div_96 = sub_150 = None
    mul_666: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_431, mul_110);  mul_110 = None
    sum_159: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_666, [0, 1]);  mul_666 = None
    sum_160: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_431, [0, 1]);  add_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_432: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_420, mul_665);  add_420 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1235: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_432, [128, 4096])
    mm_318: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1235, permute_874);  permute_874 = None
    permute_875: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1235, [1, 0])
    mm_319: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_875, view_308);  view_308 = None
    permute_876: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_319, [1, 0]);  mm_319 = None
    sum_161: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1235, [0], True)
    view_1236: "f32[4096]" = torch.ops.aten.reshape.default(sum_161, [4096]);  sum_161 = None
    permute_877: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
    view_1237: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_318, [1, 128, 16384]);  mm_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_667: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1237, mul_106);  mul_106 = None
    mul_668: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1237, add_85);  view_1237 = add_85 = None
    mul_669: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
    sub_151: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_669);  mul_669 = None
    mul_670: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_667, sub_151);  mul_667 = sub_151 = None
    mul_671: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_670, 0.7978845608028654);  mul_670 = None
    mul_672: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_671, 0.044715)
    pow_46: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 2.0);  view_307 = None
    mul_673: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
    mul_674: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_433: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_671, mul_674);  mul_671 = mul_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_675: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_668, 0.5);  mul_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_434: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_433, mul_675);  add_433 = mul_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1238: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_434, [128, 16384]);  add_434 = None
    mm_320: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1238, permute_878);  permute_878 = None
    permute_879: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1238, [1, 0])
    mm_321: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_879, view_282);  permute_879 = None
    permute_880: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
    sum_162: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1238, [0], True);  view_1238 = None
    view_1239: "f32[16384]" = torch.ops.aten.reshape.default(sum_162, [16384]);  sum_162 = None
    permute_881: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_880, [1, 0]);  permute_880 = None
    view_1240: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_320, [1, 128, 4096]);  mm_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_322: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_875, view_304);  permute_875 = view_304 = None
    permute_883: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_322, [1, 0]);  mm_322 = None
    mm_323: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1235, permute_884);  view_1235 = permute_884 = None
    view_1242: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_323, [1, 128, 4096]);  mm_323 = None
    permute_885: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_883, [1, 0]);  permute_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1243: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1242, [1, 128, 16, 256]);  view_1242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_886: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1243, [0, 2, 1, 3]);  view_1243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1244: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_886, [16, 128, 256]);  permute_886 = None
    bmm_124: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_887, view_1244);  permute_887 = None
    bmm_125: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1244, permute_888);  view_1244 = permute_888 = None
    view_1245: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_124, [1, 16, 128, 256]);  bmm_124 = None
    view_1246: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_676: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1246, alias_95);  view_1246 = None
    sum_163: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_676, [-1], True)
    mul_677: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_163);  alias_95 = sum_163 = None
    sub_152: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_97: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_152, primals_318);  sub_152 = primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_53: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_528, div_97, full_default_29);  slice_528 = div_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1247: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_53, [16, 128, 128]);  where_53 = None
    bmm_126: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_889, view_1247);  permute_889 = None
    bmm_127: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1247, permute_890);  view_1247 = permute_890 = None
    view_1248: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_126, [1, 16, 256, 128]);  bmm_126 = None
    view_1249: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_127, [1, 16, 128, 256]);  bmm_127 = None
    permute_891: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1248, [0, 1, 3, 2]);  view_1248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_892: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1249, [0, 2, 1, 3]);  view_1249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_893: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_891, [0, 2, 1, 3]);  permute_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1413: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_892, 3, 0, 64)
    slice_1414: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_892, 3, 64, 256);  permute_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1415: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_893, 3, 0, 64)
    slice_1416: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_893, 3, 64, 256);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_678: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1413, view_291)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1250: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_678, [1, 128, 16, 32, 2]);  mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_68: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1250, 4, 0)
    select_69: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1250, 4, 1);  view_1250 = None
    neg_92: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_68);  select_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_544: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_92, 3, 1, 9223372036854775807, 2);  neg_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_548: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_435: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_544, slice_scatter_548);  slice_scatter_544 = slice_scatter_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_679: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1413, view_292);  slice_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_436: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_435, mul_679);  add_435 = mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_680: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1415, view_291);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1251: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_680, [1, 128, 16, 32, 2]);  mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_70: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1251, 4, 0)
    select_71: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1251, 4, 1);  view_1251 = None
    neg_93: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_70);  select_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_552: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_93, 3, 1, 9223372036854775807, 2);  neg_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_556: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_437: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_552, slice_scatter_556);  slice_scatter_552 = slice_scatter_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_681: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1415, view_292);  slice_1415 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_438: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_437, mul_681);  add_437 = mul_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_560: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1414, 3, 64, 9223372036854775807);  slice_1414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_564: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_436, 3, 0, 64);  add_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_439: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_560, slice_scatter_564);  slice_scatter_560 = slice_scatter_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_568: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1416, 3, 64, 9223372036854775807);  slice_1416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_572: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_438, 3, 0, 64);  add_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_440: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_568, slice_scatter_572);  slice_scatter_568 = slice_scatter_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_894: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1245, [0, 2, 1, 3]);  view_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_244: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
    view_1252: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_244, [1, 128, 4096]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1253: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_440, [1, 128, 4096]);  add_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1254: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_439, [1, 128, 4096]);  add_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1255: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1252, [128, 4096]);  view_1252 = None
    permute_895: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1255, [1, 0])
    mm_324: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_895, view_282);  permute_895 = None
    permute_896: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_324, [1, 0]);  mm_324 = None
    mm_325: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1255, permute_897);  view_1255 = permute_897 = None
    view_1256: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_325, [1, 128, 4096]);  mm_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_441: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1240, view_1256);  view_1240 = view_1256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_898: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_896, [1, 0]);  permute_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1257: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1253, [128, 4096]);  view_1253 = None
    permute_899: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1257, [1, 0])
    mm_326: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_899, view_282);  permute_899 = None
    permute_900: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_326, [1, 0]);  mm_326 = None
    mm_327: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1257, permute_901);  view_1257 = permute_901 = None
    view_1258: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_442: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_441, view_1258);  add_441 = view_1258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_902: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_900, [1, 0]);  permute_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1259: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1254, [128, 4096]);  view_1254 = None
    permute_903: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1259, [1, 0])
    mm_328: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_903, view_282);  permute_903 = view_282 = None
    permute_904: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_328, [1, 0]);  mm_328 = None
    mm_329: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1259, permute_905);  view_1259 = permute_905 = None
    view_1260: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_329, [1, 128, 4096]);  mm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_443: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_442, view_1260);  add_442 = view_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_906: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_904, [1, 0]);  permute_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_683: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_443, primals_102);  primals_102 = None
    mul_684: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_683, 4096)
    sum_164: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [2], True)
    mul_685: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_683, mul_100);  mul_683 = None
    sum_165: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_685, [2], True);  mul_685 = None
    mul_686: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, sum_165);  sum_165 = None
    sub_154: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_684, sum_164);  mul_684 = sum_164 = None
    sub_155: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_154, mul_686);  sub_154 = mul_686 = None
    mul_687: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_98, sub_155);  div_98 = sub_155 = None
    mul_688: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_443, mul_100);  mul_100 = None
    sum_166: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 1]);  mul_688 = None
    sum_167: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_443, [0, 1]);  add_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_444: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_432, mul_687);  add_432 = mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1261: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_444, [128, 4096])
    mm_330: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1261, permute_907);  permute_907 = None
    permute_908: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1261, [1, 0])
    mm_331: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_908, view_280);  view_280 = None
    permute_909: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
    sum_168: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1261, [0], True)
    view_1262: "f32[4096]" = torch.ops.aten.reshape.default(sum_168, [4096]);  sum_168 = None
    permute_910: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_909, [1, 0]);  permute_909 = None
    view_1263: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_330, [1, 128, 16384]);  mm_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_689: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1263, mul_96);  mul_96 = None
    mul_690: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1263, add_77);  view_1263 = add_77 = None
    mul_691: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
    sub_156: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_691);  mul_691 = None
    mul_692: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_689, sub_156);  mul_689 = sub_156 = None
    mul_693: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_692, 0.7978845608028654);  mul_692 = None
    mul_694: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_693, 0.044715)
    pow_47: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_279, 2.0);  view_279 = None
    mul_695: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
    mul_696: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_445: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_693, mul_696);  mul_693 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_697: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_690, 0.5);  mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_446: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_445, mul_697);  add_445 = mul_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1264: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_446, [128, 16384]);  add_446 = None
    mm_332: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1264, permute_911);  permute_911 = None
    permute_912: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1264, [1, 0])
    mm_333: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_912, view_254);  permute_912 = None
    permute_913: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
    sum_169: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1264, [0], True);  view_1264 = None
    view_1265: "f32[16384]" = torch.ops.aten.reshape.default(sum_169, [16384]);  sum_169 = None
    permute_914: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
    view_1266: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_332, [1, 128, 4096]);  mm_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_334: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_908, view_276);  permute_908 = view_276 = None
    permute_916: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_334, [1, 0]);  mm_334 = None
    mm_335: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1261, permute_917);  view_1261 = permute_917 = None
    view_1268: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_335, [1, 128, 4096]);  mm_335 = None
    permute_918: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_916, [1, 0]);  permute_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1269: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1268, [1, 128, 16, 256]);  view_1268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_919: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1269, [0, 2, 1, 3]);  view_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1270: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_919, [16, 128, 256]);  permute_919 = None
    bmm_128: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_920, view_1270);  permute_920 = None
    bmm_129: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1270, permute_921);  view_1270 = permute_921 = None
    view_1271: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_128, [1, 16, 128, 256]);  bmm_128 = None
    view_1272: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_698: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1272, alias_97);  view_1272 = None
    sum_170: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [-1], True)
    mul_699: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_97, sum_170);  alias_97 = sum_170 = None
    sub_157: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_99: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_157, primals_315);  sub_157 = primals_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_54: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_480, div_99, full_default_29);  slice_480 = div_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1273: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_54, [16, 128, 128]);  where_54 = None
    bmm_130: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_922, view_1273);  permute_922 = None
    bmm_131: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1273, permute_923);  view_1273 = permute_923 = None
    view_1274: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_130, [1, 16, 256, 128]);  bmm_130 = None
    view_1275: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_131, [1, 16, 128, 256]);  bmm_131 = None
    permute_924: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1274, [0, 1, 3, 2]);  view_1274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_925: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1275, [0, 2, 1, 3]);  view_1275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_926: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_924, [0, 2, 1, 3]);  permute_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1417: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_925, 3, 0, 64)
    slice_1418: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_925, 3, 64, 256);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1419: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_926, 3, 0, 64)
    slice_1420: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_926, 3, 64, 256);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_700: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1417, view_263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1276: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_700, [1, 128, 16, 32, 2]);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_72: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1276, 4, 0)
    select_73: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1276, 4, 1);  view_1276 = None
    neg_94: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_72);  select_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_576: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_94, 3, 1, 9223372036854775807, 2);  neg_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_580: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_447: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_576, slice_scatter_580);  slice_scatter_576 = slice_scatter_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_701: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1417, view_264);  slice_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_448: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_447, mul_701);  add_447 = mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_702: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1419, view_263);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1277: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_702, [1, 128, 16, 32, 2]);  mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_74: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1277, 4, 0)
    select_75: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1277, 4, 1);  view_1277 = None
    neg_95: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_74);  select_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_584: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_95, 3, 1, 9223372036854775807, 2);  neg_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_588: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_449: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_584, slice_scatter_588);  slice_scatter_584 = slice_scatter_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_703: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1419, view_264);  slice_1419 = view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_450: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_449, mul_703);  add_449 = mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_592: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1418, 3, 64, 9223372036854775807);  slice_1418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_596: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_448, 3, 0, 64);  add_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_451: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_592, slice_scatter_596);  slice_scatter_592 = slice_scatter_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_600: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1420, 3, 64, 9223372036854775807);  slice_1420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_604: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_450, 3, 0, 64);  add_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_452: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_600, slice_scatter_604);  slice_scatter_600 = slice_scatter_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_927: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1271, [0, 2, 1, 3]);  view_1271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_245: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_927, memory_format = torch.contiguous_format);  permute_927 = None
    view_1278: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_245, [1, 128, 4096]);  clone_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1279: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_452, [1, 128, 4096]);  add_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1280: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_451, [1, 128, 4096]);  add_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1281: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1278, [128, 4096]);  view_1278 = None
    permute_928: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1281, [1, 0])
    mm_336: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_928, view_254);  permute_928 = None
    permute_929: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_336, [1, 0]);  mm_336 = None
    mm_337: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1281, permute_930);  view_1281 = permute_930 = None
    view_1282: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_337, [1, 128, 4096]);  mm_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_453: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1266, view_1282);  view_1266 = view_1282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_931: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_929, [1, 0]);  permute_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1283: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1279, [128, 4096]);  view_1279 = None
    permute_932: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1283, [1, 0])
    mm_338: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_932, view_254);  permute_932 = None
    permute_933: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_338, [1, 0]);  mm_338 = None
    mm_339: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1283, permute_934);  view_1283 = permute_934 = None
    view_1284: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_454: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_453, view_1284);  add_453 = view_1284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_935: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_933, [1, 0]);  permute_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1285: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1280, [128, 4096]);  view_1280 = None
    permute_936: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1285, [1, 0])
    mm_340: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_936, view_254);  permute_936 = view_254 = None
    permute_937: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_340, [1, 0]);  mm_340 = None
    mm_341: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1285, permute_938);  view_1285 = permute_938 = None
    view_1286: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_341, [1, 128, 4096]);  mm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_455: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_454, view_1286);  add_454 = view_1286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_939: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_937, [1, 0]);  permute_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_705: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_455, primals_92);  primals_92 = None
    mul_706: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_705, 4096)
    sum_171: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
    mul_707: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_705, mul_90);  mul_705 = None
    sum_172: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
    mul_708: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, sum_172);  sum_172 = None
    sub_159: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_706, sum_171);  mul_706 = sum_171 = None
    sub_160: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_159, mul_708);  sub_159 = mul_708 = None
    mul_709: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_100, sub_160);  div_100 = sub_160 = None
    mul_710: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_455, mul_90);  mul_90 = None
    sum_173: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
    sum_174: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_455, [0, 1]);  add_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_456: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_444, mul_709);  add_444 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1287: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_456, [128, 4096])
    mm_342: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1287, permute_940);  permute_940 = None
    permute_941: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1287, [1, 0])
    mm_343: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_941, view_252);  view_252 = None
    permute_942: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_343, [1, 0]);  mm_343 = None
    sum_175: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1287, [0], True)
    view_1288: "f32[4096]" = torch.ops.aten.reshape.default(sum_175, [4096]);  sum_175 = None
    permute_943: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_942, [1, 0]);  permute_942 = None
    view_1289: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_342, [1, 128, 16384]);  mm_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_711: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1289, mul_86);  mul_86 = None
    mul_712: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1289, add_69);  view_1289 = add_69 = None
    mul_713: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
    sub_161: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_713);  mul_713 = None
    mul_714: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_711, sub_161);  mul_711 = sub_161 = None
    mul_715: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_714, 0.7978845608028654);  mul_714 = None
    mul_716: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_715, 0.044715)
    pow_48: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 2.0);  view_251 = None
    mul_717: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
    mul_718: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_457: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_715, mul_718);  mul_715 = mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_719: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_712, 0.5);  mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_458: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_457, mul_719);  add_457 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1290: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_458, [128, 16384]);  add_458 = None
    mm_344: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1290, permute_944);  permute_944 = None
    permute_945: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1290, [1, 0])
    mm_345: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_945, view_226);  permute_945 = None
    permute_946: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
    sum_176: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1290, [0], True);  view_1290 = None
    view_1291: "f32[16384]" = torch.ops.aten.reshape.default(sum_176, [16384]);  sum_176 = None
    permute_947: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_946, [1, 0]);  permute_946 = None
    view_1292: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_344, [1, 128, 4096]);  mm_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_346: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_941, view_248);  permute_941 = view_248 = None
    permute_949: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_346, [1, 0]);  mm_346 = None
    mm_347: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1287, permute_950);  view_1287 = permute_950 = None
    view_1294: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_347, [1, 128, 4096]);  mm_347 = None
    permute_951: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_949, [1, 0]);  permute_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1295: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1294, [1, 128, 16, 256]);  view_1294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_952: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1295, [0, 2, 1, 3]);  view_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1296: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_952, [16, 128, 256]);  permute_952 = None
    bmm_132: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_953, view_1296);  permute_953 = None
    bmm_133: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1296, permute_954);  view_1296 = permute_954 = None
    view_1297: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_132, [1, 16, 128, 256]);  bmm_132 = None
    view_1298: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_720: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1298, alias_99);  view_1298 = None
    sum_177: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [-1], True)
    mul_721: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_99, sum_177);  alias_99 = sum_177 = None
    sub_162: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_101: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_162, primals_312);  sub_162 = primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_55: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_432, div_101, full_default_29);  slice_432 = div_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1299: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_55, [16, 128, 128]);  where_55 = None
    bmm_134: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_955, view_1299);  permute_955 = None
    bmm_135: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1299, permute_956);  view_1299 = permute_956 = None
    view_1300: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_134, [1, 16, 256, 128]);  bmm_134 = None
    view_1301: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_135, [1, 16, 128, 256]);  bmm_135 = None
    permute_957: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1300, [0, 1, 3, 2]);  view_1300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_958: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1301, [0, 2, 1, 3]);  view_1301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_959: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_957, [0, 2, 1, 3]);  permute_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1421: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_958, 3, 0, 64)
    slice_1422: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_958, 3, 64, 256);  permute_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1423: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_959, 3, 0, 64)
    slice_1424: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_959, 3, 64, 256);  permute_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_722: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1421, view_235)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1302: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_722, [1, 128, 16, 32, 2]);  mul_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_76: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1302, 4, 0)
    select_77: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1302, 4, 1);  view_1302 = None
    neg_96: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_76);  select_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_608: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_96, 3, 1, 9223372036854775807, 2);  neg_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_612: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_459: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_608, slice_scatter_612);  slice_scatter_608 = slice_scatter_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_723: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1421, view_236);  slice_1421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_460: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_459, mul_723);  add_459 = mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_724: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1423, view_235);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1303: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_724, [1, 128, 16, 32, 2]);  mul_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_78: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1303, 4, 0)
    select_79: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1303, 4, 1);  view_1303 = None
    neg_97: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_78);  select_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_616: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_97, 3, 1, 9223372036854775807, 2);  neg_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_620: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_461: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_616, slice_scatter_620);  slice_scatter_616 = slice_scatter_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_725: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1423, view_236);  slice_1423 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_462: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_461, mul_725);  add_461 = mul_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_624: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1422, 3, 64, 9223372036854775807);  slice_1422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_628: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_460, 3, 0, 64);  add_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_463: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_624, slice_scatter_628);  slice_scatter_624 = slice_scatter_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_632: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1424, 3, 64, 9223372036854775807);  slice_1424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_636: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_462, 3, 0, 64);  add_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_464: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_632, slice_scatter_636);  slice_scatter_632 = slice_scatter_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_960: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1297, [0, 2, 1, 3]);  view_1297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_246: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_960, memory_format = torch.contiguous_format);  permute_960 = None
    view_1304: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_246, [1, 128, 4096]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1305: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_464, [1, 128, 4096]);  add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1306: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_463, [1, 128, 4096]);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1307: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1304, [128, 4096]);  view_1304 = None
    permute_961: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1307, [1, 0])
    mm_348: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_961, view_226);  permute_961 = None
    permute_962: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_348, [1, 0]);  mm_348 = None
    mm_349: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1307, permute_963);  view_1307 = permute_963 = None
    view_1308: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_349, [1, 128, 4096]);  mm_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_465: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1292, view_1308);  view_1292 = view_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_964: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_962, [1, 0]);  permute_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1309: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1305, [128, 4096]);  view_1305 = None
    permute_965: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1309, [1, 0])
    mm_350: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_965, view_226);  permute_965 = None
    permute_966: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_350, [1, 0]);  mm_350 = None
    mm_351: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1309, permute_967);  view_1309 = permute_967 = None
    view_1310: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_466: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_465, view_1310);  add_465 = view_1310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_968: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_966, [1, 0]);  permute_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1311: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1306, [128, 4096]);  view_1306 = None
    permute_969: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1311, [1, 0])
    mm_352: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_969, view_226);  permute_969 = view_226 = None
    permute_970: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_352, [1, 0]);  mm_352 = None
    mm_353: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1311, permute_971);  view_1311 = permute_971 = None
    view_1312: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_353, [1, 128, 4096]);  mm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_467: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_466, view_1312);  add_466 = view_1312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_972: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_970, [1, 0]);  permute_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_727: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_467, primals_82);  primals_82 = None
    mul_728: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_727, 4096)
    sum_178: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True)
    mul_729: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_727, mul_80);  mul_727 = None
    sum_179: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_729, [2], True);  mul_729 = None
    mul_730: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, sum_179);  sum_179 = None
    sub_164: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_728, sum_178);  mul_728 = sum_178 = None
    sub_165: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_164, mul_730);  sub_164 = mul_730 = None
    mul_731: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_102, sub_165);  div_102 = sub_165 = None
    mul_732: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_467, mul_80);  mul_80 = None
    sum_180: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 1]);  mul_732 = None
    sum_181: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_467, [0, 1]);  add_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_468: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_456, mul_731);  add_456 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1313: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_468, [128, 4096])
    mm_354: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1313, permute_973);  permute_973 = None
    permute_974: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1313, [1, 0])
    mm_355: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_974, view_224);  view_224 = None
    permute_975: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_355, [1, 0]);  mm_355 = None
    sum_182: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1313, [0], True)
    view_1314: "f32[4096]" = torch.ops.aten.reshape.default(sum_182, [4096]);  sum_182 = None
    permute_976: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_975, [1, 0]);  permute_975 = None
    view_1315: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_354, [1, 128, 16384]);  mm_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_733: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1315, mul_76);  mul_76 = None
    mul_734: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1315, add_61);  view_1315 = add_61 = None
    mul_735: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
    sub_166: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_735);  mul_735 = None
    mul_736: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_733, sub_166);  mul_733 = sub_166 = None
    mul_737: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_736, 0.7978845608028654);  mul_736 = None
    mul_738: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_737, 0.044715)
    pow_49: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_223, 2.0);  view_223 = None
    mul_739: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
    mul_740: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_738, mul_739);  mul_738 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_469: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_737, mul_740);  mul_737 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_741: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_734, 0.5);  mul_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_470: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_469, mul_741);  add_469 = mul_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1316: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_470, [128, 16384]);  add_470 = None
    mm_356: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1316, permute_977);  permute_977 = None
    permute_978: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1316, [1, 0])
    mm_357: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_978, view_198);  permute_978 = None
    permute_979: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
    sum_183: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1316, [0], True);  view_1316 = None
    view_1317: "f32[16384]" = torch.ops.aten.reshape.default(sum_183, [16384]);  sum_183 = None
    permute_980: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_979, [1, 0]);  permute_979 = None
    view_1318: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_356, [1, 128, 4096]);  mm_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_358: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_974, view_220);  permute_974 = view_220 = None
    permute_982: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_358, [1, 0]);  mm_358 = None
    mm_359: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1313, permute_983);  view_1313 = permute_983 = None
    view_1320: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_359, [1, 128, 4096]);  mm_359 = None
    permute_984: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_982, [1, 0]);  permute_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1321: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1320, [1, 128, 16, 256]);  view_1320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_985: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1321, [0, 2, 1, 3]);  view_1321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1322: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_985, [16, 128, 256]);  permute_985 = None
    bmm_136: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_986, view_1322);  permute_986 = None
    bmm_137: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1322, permute_987);  view_1322 = permute_987 = None
    view_1323: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_136, [1, 16, 128, 256]);  bmm_136 = None
    view_1324: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_742: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1324, alias_101);  view_1324 = None
    sum_184: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [-1], True)
    mul_743: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_101, sum_184);  alias_101 = sum_184 = None
    sub_167: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_103: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_167, primals_309);  sub_167 = primals_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_56: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_384, div_103, full_default_29);  slice_384 = div_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1325: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_56, [16, 128, 128]);  where_56 = None
    bmm_138: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_988, view_1325);  permute_988 = None
    bmm_139: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1325, permute_989);  view_1325 = permute_989 = None
    view_1326: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_138, [1, 16, 256, 128]);  bmm_138 = None
    view_1327: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_139, [1, 16, 128, 256]);  bmm_139 = None
    permute_990: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1326, [0, 1, 3, 2]);  view_1326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_991: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1327, [0, 2, 1, 3]);  view_1327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_992: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_990, [0, 2, 1, 3]);  permute_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1425: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_991, 3, 0, 64)
    slice_1426: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_991, 3, 64, 256);  permute_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1427: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_992, 3, 0, 64)
    slice_1428: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_992, 3, 64, 256);  permute_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_744: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1425, view_207)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1328: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_744, [1, 128, 16, 32, 2]);  mul_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_80: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1328, 4, 0)
    select_81: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1328, 4, 1);  view_1328 = None
    neg_98: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_80);  select_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_640: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_98, 3, 1, 9223372036854775807, 2);  neg_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_644: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_471: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_640, slice_scatter_644);  slice_scatter_640 = slice_scatter_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_745: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1425, view_208);  slice_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_472: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_471, mul_745);  add_471 = mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_746: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1427, view_207);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1329: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_746, [1, 128, 16, 32, 2]);  mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_82: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1329, 4, 0)
    select_83: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1329, 4, 1);  view_1329 = None
    neg_99: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_82);  select_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_648: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_99, 3, 1, 9223372036854775807, 2);  neg_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_652: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_473: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_648, slice_scatter_652);  slice_scatter_648 = slice_scatter_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_747: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1427, view_208);  slice_1427 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_474: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_473, mul_747);  add_473 = mul_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_656: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1426, 3, 64, 9223372036854775807);  slice_1426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_660: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_472, 3, 0, 64);  add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_475: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_656, slice_scatter_660);  slice_scatter_656 = slice_scatter_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_664: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1428, 3, 64, 9223372036854775807);  slice_1428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_668: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_474, 3, 0, 64);  add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_476: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_664, slice_scatter_668);  slice_scatter_664 = slice_scatter_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_993: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1323, [0, 2, 1, 3]);  view_1323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_247: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_993, memory_format = torch.contiguous_format);  permute_993 = None
    view_1330: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_247, [1, 128, 4096]);  clone_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1331: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_476, [1, 128, 4096]);  add_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1332: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_475, [1, 128, 4096]);  add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1333: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1330, [128, 4096]);  view_1330 = None
    permute_994: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1333, [1, 0])
    mm_360: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_994, view_198);  permute_994 = None
    permute_995: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_360, [1, 0]);  mm_360 = None
    mm_361: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1333, permute_996);  view_1333 = permute_996 = None
    view_1334: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_361, [1, 128, 4096]);  mm_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_477: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1318, view_1334);  view_1318 = view_1334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_997: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_995, [1, 0]);  permute_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1335: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1331, [128, 4096]);  view_1331 = None
    permute_998: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1335, [1, 0])
    mm_362: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_998, view_198);  permute_998 = None
    permute_999: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_362, [1, 0]);  mm_362 = None
    mm_363: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1335, permute_1000);  view_1335 = permute_1000 = None
    view_1336: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_478: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_477, view_1336);  add_477 = view_1336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1001: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_999, [1, 0]);  permute_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1337: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1332, [128, 4096]);  view_1332 = None
    permute_1002: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1337, [1, 0])
    mm_364: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1002, view_198);  permute_1002 = view_198 = None
    permute_1003: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_364, [1, 0]);  mm_364 = None
    mm_365: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1337, permute_1004);  view_1337 = permute_1004 = None
    view_1338: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_365, [1, 128, 4096]);  mm_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_479: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_478, view_1338);  add_478 = view_1338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1005: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_749: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_479, primals_72);  primals_72 = None
    mul_750: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_749, 4096)
    sum_185: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
    mul_751: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_749, mul_70);  mul_749 = None
    sum_186: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
    mul_752: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, sum_186);  sum_186 = None
    sub_169: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_750, sum_185);  mul_750 = sum_185 = None
    sub_170: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_169, mul_752);  sub_169 = mul_752 = None
    mul_753: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_104, sub_170);  div_104 = sub_170 = None
    mul_754: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_479, mul_70);  mul_70 = None
    sum_187: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
    sum_188: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_479, [0, 1]);  add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_480: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_468, mul_753);  add_468 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1339: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_480, [128, 4096])
    mm_366: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1339, permute_1006);  permute_1006 = None
    permute_1007: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1339, [1, 0])
    mm_367: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1007, view_196);  view_196 = None
    permute_1008: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_367, [1, 0]);  mm_367 = None
    sum_189: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1339, [0], True)
    view_1340: "f32[4096]" = torch.ops.aten.reshape.default(sum_189, [4096]);  sum_189 = None
    permute_1009: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1008, [1, 0]);  permute_1008 = None
    view_1341: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_366, [1, 128, 16384]);  mm_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_755: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1341, mul_66);  mul_66 = None
    mul_756: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1341, add_53);  view_1341 = add_53 = None
    mul_757: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
    sub_171: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_757);  mul_757 = None
    mul_758: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_755, sub_171);  mul_755 = sub_171 = None
    mul_759: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_758, 0.7978845608028654);  mul_758 = None
    mul_760: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_759, 0.044715)
    pow_50: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 2.0);  view_195 = None
    mul_761: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
    mul_762: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_481: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_759, mul_762);  mul_759 = mul_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_763: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_756, 0.5);  mul_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_482: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_481, mul_763);  add_481 = mul_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1342: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_482, [128, 16384]);  add_482 = None
    mm_368: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1342, permute_1010);  permute_1010 = None
    permute_1011: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1342, [1, 0])
    mm_369: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1011, view_170);  permute_1011 = None
    permute_1012: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_369, [1, 0]);  mm_369 = None
    sum_190: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1342, [0], True);  view_1342 = None
    view_1343: "f32[16384]" = torch.ops.aten.reshape.default(sum_190, [16384]);  sum_190 = None
    permute_1013: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1012, [1, 0]);  permute_1012 = None
    view_1344: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_368, [1, 128, 4096]);  mm_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_370: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1007, view_192);  permute_1007 = view_192 = None
    permute_1015: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_370, [1, 0]);  mm_370 = None
    mm_371: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1339, permute_1016);  view_1339 = permute_1016 = None
    view_1346: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_371, [1, 128, 4096]);  mm_371 = None
    permute_1017: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1015, [1, 0]);  permute_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1347: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1346, [1, 128, 16, 256]);  view_1346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1018: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1347, [0, 2, 1, 3]);  view_1347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1348: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1018, [16, 128, 256]);  permute_1018 = None
    bmm_140: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1019, view_1348);  permute_1019 = None
    bmm_141: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1348, permute_1020);  view_1348 = permute_1020 = None
    view_1349: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_140, [1, 16, 128, 256]);  bmm_140 = None
    view_1350: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_764: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1350, alias_103);  view_1350 = None
    sum_191: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_764, [-1], True)
    mul_765: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_103, sum_191);  alias_103 = sum_191 = None
    sub_172: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_105: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_172, primals_306);  sub_172 = primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_57: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_336, div_105, full_default_29);  slice_336 = div_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1351: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_57, [16, 128, 128]);  where_57 = None
    bmm_142: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1021, view_1351);  permute_1021 = None
    bmm_143: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1351, permute_1022);  view_1351 = permute_1022 = None
    view_1352: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_142, [1, 16, 256, 128]);  bmm_142 = None
    view_1353: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_143, [1, 16, 128, 256]);  bmm_143 = None
    permute_1023: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1352, [0, 1, 3, 2]);  view_1352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1024: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1353, [0, 2, 1, 3]);  view_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1025: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1023, [0, 2, 1, 3]);  permute_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1429: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 0, 64)
    slice_1430: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 64, 256);  permute_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1431: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 0, 64)
    slice_1432: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 64, 256);  permute_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_766: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1429, view_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1354: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_766, [1, 128, 16, 32, 2]);  mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_84: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1354, 4, 0)
    select_85: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1354, 4, 1);  view_1354 = None
    neg_100: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_84);  select_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_672: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_100, 3, 1, 9223372036854775807, 2);  neg_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_676: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_483: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_672, slice_scatter_676);  slice_scatter_672 = slice_scatter_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_767: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1429, view_180);  slice_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_484: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_483, mul_767);  add_483 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_768: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1431, view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1355: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_768, [1, 128, 16, 32, 2]);  mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_86: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1355, 4, 0)
    select_87: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1355, 4, 1);  view_1355 = None
    neg_101: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_86);  select_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_680: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_101, 3, 1, 9223372036854775807, 2);  neg_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_684: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_485: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_680, slice_scatter_684);  slice_scatter_680 = slice_scatter_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_769: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1431, view_180);  slice_1431 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_486: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_485, mul_769);  add_485 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_688: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1430, 3, 64, 9223372036854775807);  slice_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_692: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_484, 3, 0, 64);  add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_487: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_688, slice_scatter_692);  slice_scatter_688 = slice_scatter_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_696: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1432, 3, 64, 9223372036854775807);  slice_1432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_700: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_486, 3, 0, 64);  add_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_488: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_696, slice_scatter_700);  slice_scatter_696 = slice_scatter_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1026: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1349, [0, 2, 1, 3]);  view_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_248: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
    view_1356: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_248, [1, 128, 4096]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1357: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_488, [1, 128, 4096]);  add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1358: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_487, [1, 128, 4096]);  add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1359: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1356, [128, 4096]);  view_1356 = None
    permute_1027: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1359, [1, 0])
    mm_372: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1027, view_170);  permute_1027 = None
    permute_1028: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_372, [1, 0]);  mm_372 = None
    mm_373: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1359, permute_1029);  view_1359 = permute_1029 = None
    view_1360: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_373, [1, 128, 4096]);  mm_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_489: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1344, view_1360);  view_1344 = view_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1030: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1361: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1357, [128, 4096]);  view_1357 = None
    permute_1031: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1361, [1, 0])
    mm_374: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1031, view_170);  permute_1031 = None
    permute_1032: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_374, [1, 0]);  mm_374 = None
    mm_375: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1361, permute_1033);  view_1361 = permute_1033 = None
    view_1362: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_490: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_489, view_1362);  add_489 = view_1362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1034: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1032, [1, 0]);  permute_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1363: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1358, [128, 4096]);  view_1358 = None
    permute_1035: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1363, [1, 0])
    mm_376: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1035, view_170);  permute_1035 = view_170 = None
    permute_1036: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_376, [1, 0]);  mm_376 = None
    mm_377: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1363, permute_1037);  view_1363 = permute_1037 = None
    view_1364: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_377, [1, 128, 4096]);  mm_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_491: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_490, view_1364);  add_490 = view_1364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1038: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1036, [1, 0]);  permute_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_771: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_491, primals_62);  primals_62 = None
    mul_772: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_771, 4096)
    sum_192: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [2], True)
    mul_773: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_771, mul_60);  mul_771 = None
    sum_193: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True);  mul_773 = None
    mul_774: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, sum_193);  sum_193 = None
    sub_174: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_772, sum_192);  mul_772 = sum_192 = None
    sub_175: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_174, mul_774);  sub_174 = mul_774 = None
    mul_775: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_106, sub_175);  div_106 = sub_175 = None
    mul_776: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_491, mul_60);  mul_60 = None
    sum_194: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_776, [0, 1]);  mul_776 = None
    sum_195: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_491, [0, 1]);  add_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_492: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_480, mul_775);  add_480 = mul_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1365: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_492, [128, 4096])
    mm_378: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1365, permute_1039);  permute_1039 = None
    permute_1040: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1365, [1, 0])
    mm_379: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1040, view_168);  view_168 = None
    permute_1041: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_379, [1, 0]);  mm_379 = None
    sum_196: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1365, [0], True)
    view_1366: "f32[4096]" = torch.ops.aten.reshape.default(sum_196, [4096]);  sum_196 = None
    permute_1042: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1041, [1, 0]);  permute_1041 = None
    view_1367: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_378, [1, 128, 16384]);  mm_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_777: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1367, mul_56);  mul_56 = None
    mul_778: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1367, add_45);  view_1367 = add_45 = None
    mul_779: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
    sub_176: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_779);  mul_779 = None
    mul_780: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_777, sub_176);  mul_777 = sub_176 = None
    mul_781: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_780, 0.7978845608028654);  mul_780 = None
    mul_782: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_781, 0.044715)
    pow_51: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_167, 2.0);  view_167 = None
    mul_783: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
    mul_784: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_493: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_781, mul_784);  mul_781 = mul_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_785: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_778, 0.5);  mul_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_494: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_493, mul_785);  add_493 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1368: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_494, [128, 16384]);  add_494 = None
    mm_380: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1368, permute_1043);  permute_1043 = None
    permute_1044: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1368, [1, 0])
    mm_381: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1044, view_142);  permute_1044 = None
    permute_1045: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_381, [1, 0]);  mm_381 = None
    sum_197: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1368, [0], True);  view_1368 = None
    view_1369: "f32[16384]" = torch.ops.aten.reshape.default(sum_197, [16384]);  sum_197 = None
    permute_1046: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1045, [1, 0]);  permute_1045 = None
    view_1370: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_380, [1, 128, 4096]);  mm_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_382: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1040, view_164);  permute_1040 = view_164 = None
    permute_1048: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_382, [1, 0]);  mm_382 = None
    mm_383: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1365, permute_1049);  view_1365 = permute_1049 = None
    view_1372: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_383, [1, 128, 4096]);  mm_383 = None
    permute_1050: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1048, [1, 0]);  permute_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1373: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1372, [1, 128, 16, 256]);  view_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1051: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1373, [0, 2, 1, 3]);  view_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1374: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1051, [16, 128, 256]);  permute_1051 = None
    bmm_144: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1052, view_1374);  permute_1052 = None
    bmm_145: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1374, permute_1053);  view_1374 = permute_1053 = None
    view_1375: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_144, [1, 16, 128, 256]);  bmm_144 = None
    view_1376: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_145, [1, 16, 128, 128]);  bmm_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_786: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1376, alias_105);  view_1376 = None
    sum_198: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [-1], True)
    mul_787: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_105, sum_198);  alias_105 = sum_198 = None
    sub_177: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_786, mul_787);  mul_786 = mul_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_107: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_177, primals_303);  sub_177 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_58: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_288, div_107, full_default_29);  slice_288 = div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1377: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_58, [16, 128, 128]);  where_58 = None
    bmm_146: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1054, view_1377);  permute_1054 = None
    bmm_147: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1377, permute_1055);  view_1377 = permute_1055 = None
    view_1378: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_146, [1, 16, 256, 128]);  bmm_146 = None
    view_1379: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_147, [1, 16, 128, 256]);  bmm_147 = None
    permute_1056: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1378, [0, 1, 3, 2]);  view_1378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1057: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1379, [0, 2, 1, 3]);  view_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1058: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1056, [0, 2, 1, 3]);  permute_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1433: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 0, 64)
    slice_1434: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 64, 256);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1435: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 0, 64)
    slice_1436: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 64, 256);  permute_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_788: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1433, view_151)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1380: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_788, [1, 128, 16, 32, 2]);  mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_88: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1380, 4, 0)
    select_89: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1380, 4, 1);  view_1380 = None
    neg_102: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_88);  select_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_704: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_102, 3, 1, 9223372036854775807, 2);  neg_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_708: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_495: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_704, slice_scatter_708);  slice_scatter_704 = slice_scatter_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_789: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1433, view_152);  slice_1433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_496: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_495, mul_789);  add_495 = mul_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_790: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1435, view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1381: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_790, [1, 128, 16, 32, 2]);  mul_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_90: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1381, 4, 0)
    select_91: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1381, 4, 1);  view_1381 = None
    neg_103: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_90);  select_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_712: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_103, 3, 1, 9223372036854775807, 2);  neg_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_716: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_497: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_712, slice_scatter_716);  slice_scatter_712 = slice_scatter_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_791: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1435, view_152);  slice_1435 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_498: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_497, mul_791);  add_497 = mul_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_720: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1434, 3, 64, 9223372036854775807);  slice_1434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_724: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_496, 3, 0, 64);  add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_499: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_720, slice_scatter_724);  slice_scatter_720 = slice_scatter_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_728: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1436, 3, 64, 9223372036854775807);  slice_1436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_732: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_498, 3, 0, 64);  add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_500: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_728, slice_scatter_732);  slice_scatter_728 = slice_scatter_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1059: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1375, [0, 2, 1, 3]);  view_1375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_249: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
    view_1382: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_249, [1, 128, 4096]);  clone_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1383: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_500, [1, 128, 4096]);  add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1384: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_499, [1, 128, 4096]);  add_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1385: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1382, [128, 4096]);  view_1382 = None
    permute_1060: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1385, [1, 0])
    mm_384: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1060, view_142);  permute_1060 = None
    permute_1061: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_384, [1, 0]);  mm_384 = None
    mm_385: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1385, permute_1062);  view_1385 = permute_1062 = None
    view_1386: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_385, [1, 128, 4096]);  mm_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_501: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1370, view_1386);  view_1370 = view_1386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1063: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1061, [1, 0]);  permute_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1387: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1383, [128, 4096]);  view_1383 = None
    permute_1064: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1387, [1, 0])
    mm_386: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1064, view_142);  permute_1064 = None
    permute_1065: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_386, [1, 0]);  mm_386 = None
    mm_387: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1387, permute_1066);  view_1387 = permute_1066 = None
    view_1388: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_387, [1, 128, 4096]);  mm_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_502: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_501, view_1388);  add_501 = view_1388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1067: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1065, [1, 0]);  permute_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1389: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1384, [128, 4096]);  view_1384 = None
    permute_1068: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1389, [1, 0])
    mm_388: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1068, view_142);  permute_1068 = view_142 = None
    permute_1069: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_388, [1, 0]);  mm_388 = None
    mm_389: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1389, permute_1070);  view_1389 = permute_1070 = None
    view_1390: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_389, [1, 128, 4096]);  mm_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_503: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_502, view_1390);  add_502 = view_1390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1071: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1069, [1, 0]);  permute_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_793: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_503, primals_52);  primals_52 = None
    mul_794: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_793, 4096)
    sum_199: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_793, [2], True)
    mul_795: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_793, mul_50);  mul_793 = None
    sum_200: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True);  mul_795 = None
    mul_796: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, sum_200);  sum_200 = None
    sub_179: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_794, sum_199);  mul_794 = sum_199 = None
    sub_180: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_179, mul_796);  sub_179 = mul_796 = None
    mul_797: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_108, sub_180);  div_108 = sub_180 = None
    mul_798: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_503, mul_50);  mul_50 = None
    sum_201: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 1]);  mul_798 = None
    sum_202: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_503, [0, 1]);  add_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_504: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_492, mul_797);  add_492 = mul_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1391: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_504, [128, 4096])
    mm_390: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1391, permute_1072);  permute_1072 = None
    permute_1073: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1391, [1, 0])
    mm_391: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1073, view_140);  view_140 = None
    permute_1074: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_391, [1, 0]);  mm_391 = None
    sum_203: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1391, [0], True)
    view_1392: "f32[4096]" = torch.ops.aten.reshape.default(sum_203, [4096]);  sum_203 = None
    permute_1075: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1074, [1, 0]);  permute_1074 = None
    view_1393: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_390, [1, 128, 16384]);  mm_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_799: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1393, mul_46);  mul_46 = None
    mul_800: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1393, add_37);  view_1393 = add_37 = None
    mul_801: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
    sub_181: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_801);  mul_801 = None
    mul_802: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_799, sub_181);  mul_799 = sub_181 = None
    mul_803: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_802, 0.7978845608028654);  mul_802 = None
    mul_804: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_803, 0.044715)
    pow_52: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_139, 2.0);  view_139 = None
    mul_805: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
    mul_806: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_505: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_803, mul_806);  mul_803 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_807: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_800, 0.5);  mul_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_506: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_505, mul_807);  add_505 = mul_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1394: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_506, [128, 16384]);  add_506 = None
    mm_392: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1394, permute_1076);  permute_1076 = None
    permute_1077: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1394, [1, 0])
    mm_393: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1077, view_114);  permute_1077 = None
    permute_1078: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_393, [1, 0]);  mm_393 = None
    sum_204: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True);  view_1394 = None
    view_1395: "f32[16384]" = torch.ops.aten.reshape.default(sum_204, [16384]);  sum_204 = None
    permute_1079: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1078, [1, 0]);  permute_1078 = None
    view_1396: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_392, [1, 128, 4096]);  mm_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_394: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1073, view_136);  permute_1073 = view_136 = None
    permute_1081: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_394, [1, 0]);  mm_394 = None
    mm_395: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1391, permute_1082);  view_1391 = permute_1082 = None
    view_1398: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_395, [1, 128, 4096]);  mm_395 = None
    permute_1083: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1081, [1, 0]);  permute_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1399: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1398, [1, 128, 16, 256]);  view_1398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1084: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1399, [0, 2, 1, 3]);  view_1399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1400: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1084, [16, 128, 256]);  permute_1084 = None
    bmm_148: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1085, view_1400);  permute_1085 = None
    bmm_149: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1400, permute_1086);  view_1400 = permute_1086 = None
    view_1401: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_148, [1, 16, 128, 256]);  bmm_148 = None
    view_1402: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_149, [1, 16, 128, 128]);  bmm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_808: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1402, alias_107);  view_1402 = None
    sum_205: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [-1], True)
    mul_809: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_107, sum_205);  alias_107 = sum_205 = None
    sub_182: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_109: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_182, primals_300);  sub_182 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_59: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_240, div_109, full_default_29);  slice_240 = div_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1403: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_59, [16, 128, 128]);  where_59 = None
    bmm_150: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1087, view_1403);  permute_1087 = None
    bmm_151: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1403, permute_1088);  view_1403 = permute_1088 = None
    view_1404: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_150, [1, 16, 256, 128]);  bmm_150 = None
    view_1405: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_151, [1, 16, 128, 256]);  bmm_151 = None
    permute_1089: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1404, [0, 1, 3, 2]);  view_1404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1090: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1405, [0, 2, 1, 3]);  view_1405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1091: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1089, [0, 2, 1, 3]);  permute_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1437: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 0, 64)
    slice_1438: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 64, 256);  permute_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1439: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 0, 64)
    slice_1440: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 64, 256);  permute_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_810: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1437, view_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1406: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_810, [1, 128, 16, 32, 2]);  mul_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_92: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1406, 4, 0)
    select_93: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1406, 4, 1);  view_1406 = None
    neg_104: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_92);  select_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_736: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_104, 3, 1, 9223372036854775807, 2);  neg_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_740: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_507: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_736, slice_scatter_740);  slice_scatter_736 = slice_scatter_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_811: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1437, view_124);  slice_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_508: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_507, mul_811);  add_507 = mul_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_812: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1439, view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1407: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_812, [1, 128, 16, 32, 2]);  mul_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_94: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1407, 4, 0)
    select_95: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1407, 4, 1);  view_1407 = None
    neg_105: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_94);  select_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_744: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_105, 3, 1, 9223372036854775807, 2);  neg_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_748: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_509: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_744, slice_scatter_748);  slice_scatter_744 = slice_scatter_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_813: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1439, view_124);  slice_1439 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_510: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_509, mul_813);  add_509 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_752: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1438, 3, 64, 9223372036854775807);  slice_1438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_756: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_508, 3, 0, 64);  add_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_511: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_752, slice_scatter_756);  slice_scatter_752 = slice_scatter_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_760: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1440, 3, 64, 9223372036854775807);  slice_1440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_764: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_510, 3, 0, 64);  add_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_512: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_760, slice_scatter_764);  slice_scatter_760 = slice_scatter_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1092: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1401, [0, 2, 1, 3]);  view_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_250: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
    view_1408: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_250, [1, 128, 4096]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1409: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_512, [1, 128, 4096]);  add_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1410: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_511, [1, 128, 4096]);  add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1411: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1408, [128, 4096]);  view_1408 = None
    permute_1093: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1411, [1, 0])
    mm_396: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1093, view_114);  permute_1093 = None
    permute_1094: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_396, [1, 0]);  mm_396 = None
    mm_397: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1411, permute_1095);  view_1411 = permute_1095 = None
    view_1412: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_397, [1, 128, 4096]);  mm_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_513: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1396, view_1412);  view_1396 = view_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1096: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1094, [1, 0]);  permute_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1413: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1409, [128, 4096]);  view_1409 = None
    permute_1097: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1413, [1, 0])
    mm_398: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1097, view_114);  permute_1097 = None
    permute_1098: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_398, [1, 0]);  mm_398 = None
    mm_399: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1413, permute_1099);  view_1413 = permute_1099 = None
    view_1414: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_399, [1, 128, 4096]);  mm_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_514: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_513, view_1414);  add_513 = view_1414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1100: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1098, [1, 0]);  permute_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1415: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1410, [128, 4096]);  view_1410 = None
    permute_1101: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1415, [1, 0])
    mm_400: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1101, view_114);  permute_1101 = view_114 = None
    permute_1102: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_400, [1, 0]);  mm_400 = None
    mm_401: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1415, permute_1103);  view_1415 = permute_1103 = None
    view_1416: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_401, [1, 128, 4096]);  mm_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_515: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_514, view_1416);  add_514 = view_1416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1104: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1102, [1, 0]);  permute_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_815: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_515, primals_42);  primals_42 = None
    mul_816: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_815, 4096)
    sum_206: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True)
    mul_817: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_815, mul_40);  mul_815 = None
    sum_207: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True);  mul_817 = None
    mul_818: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, sum_207);  sum_207 = None
    sub_184: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_816, sum_206);  mul_816 = sum_206 = None
    sub_185: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_184, mul_818);  sub_184 = mul_818 = None
    mul_819: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_110, sub_185);  div_110 = sub_185 = None
    mul_820: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_515, mul_40);  mul_40 = None
    sum_208: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 1]);  mul_820 = None
    sum_209: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_515, [0, 1]);  add_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_516: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_504, mul_819);  add_504 = mul_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1417: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_516, [128, 4096])
    mm_402: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1417, permute_1105);  permute_1105 = None
    permute_1106: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1417, [1, 0])
    mm_403: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1106, view_112);  view_112 = None
    permute_1107: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_403, [1, 0]);  mm_403 = None
    sum_210: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1417, [0], True)
    view_1418: "f32[4096]" = torch.ops.aten.reshape.default(sum_210, [4096]);  sum_210 = None
    permute_1108: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1107, [1, 0]);  permute_1107 = None
    view_1419: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_402, [1, 128, 16384]);  mm_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_821: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1419, mul_36);  mul_36 = None
    mul_822: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1419, add_29);  view_1419 = add_29 = None
    mul_823: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
    sub_186: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_823);  mul_823 = None
    mul_824: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_821, sub_186);  mul_821 = sub_186 = None
    mul_825: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_824, 0.7978845608028654);  mul_824 = None
    mul_826: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_825, 0.044715)
    pow_53: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_111, 2.0);  view_111 = None
    mul_827: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
    mul_828: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_517: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_825, mul_828);  mul_825 = mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_829: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_822, 0.5);  mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_518: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_517, mul_829);  add_517 = mul_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1420: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_518, [128, 16384]);  add_518 = None
    mm_404: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1420, permute_1109);  permute_1109 = None
    permute_1110: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1420, [1, 0])
    mm_405: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1110, view_86);  permute_1110 = None
    permute_1111: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_405, [1, 0]);  mm_405 = None
    sum_211: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1420, [0], True);  view_1420 = None
    view_1421: "f32[16384]" = torch.ops.aten.reshape.default(sum_211, [16384]);  sum_211 = None
    permute_1112: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1111, [1, 0]);  permute_1111 = None
    view_1422: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_404, [1, 128, 4096]);  mm_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_406: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1106, view_108);  permute_1106 = view_108 = None
    permute_1114: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_406, [1, 0]);  mm_406 = None
    mm_407: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1417, permute_1115);  view_1417 = permute_1115 = None
    view_1424: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_407, [1, 128, 4096]);  mm_407 = None
    permute_1116: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1114, [1, 0]);  permute_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1425: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1424, [1, 128, 16, 256]);  view_1424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1117: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1425, [0, 2, 1, 3]);  view_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1426: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1117, [16, 128, 256]);  permute_1117 = None
    bmm_152: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1118, view_1426);  permute_1118 = None
    bmm_153: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1426, permute_1119);  view_1426 = permute_1119 = None
    view_1427: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_152, [1, 16, 128, 256]);  bmm_152 = None
    view_1428: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_153, [1, 16, 128, 128]);  bmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_830: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1428, alias_109);  view_1428 = None
    sum_212: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_830, [-1], True)
    mul_831: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_109, sum_212);  alias_109 = sum_212 = None
    sub_187: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_111: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_187, primals_297);  sub_187 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_60: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_192, div_111, full_default_29);  slice_192 = div_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1429: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_60, [16, 128, 128]);  where_60 = None
    bmm_154: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1120, view_1429);  permute_1120 = None
    bmm_155: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1429, permute_1121);  view_1429 = permute_1121 = None
    view_1430: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_154, [1, 16, 256, 128]);  bmm_154 = None
    view_1431: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_155, [1, 16, 128, 256]);  bmm_155 = None
    permute_1122: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1430, [0, 1, 3, 2]);  view_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1123: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1431, [0, 2, 1, 3]);  view_1431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1124: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1122, [0, 2, 1, 3]);  permute_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1441: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 0, 64)
    slice_1442: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 64, 256);  permute_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1443: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 0, 64)
    slice_1444: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 64, 256);  permute_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_832: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1441, view_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1432: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_832, [1, 128, 16, 32, 2]);  mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_96: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1432, 4, 0)
    select_97: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1432, 4, 1);  view_1432 = None
    neg_106: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_96);  select_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_768: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_106, 3, 1, 9223372036854775807, 2);  neg_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_772: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_519: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_768, slice_scatter_772);  slice_scatter_768 = slice_scatter_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_833: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1441, view_96);  slice_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_520: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_519, mul_833);  add_519 = mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_834: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1443, view_95);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1433: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_834, [1, 128, 16, 32, 2]);  mul_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_98: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1433, 4, 0)
    select_99: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1433, 4, 1);  view_1433 = None
    neg_107: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_98);  select_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_776: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_107, 3, 1, 9223372036854775807, 2);  neg_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_780: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_521: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_776, slice_scatter_780);  slice_scatter_776 = slice_scatter_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_835: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1443, view_96);  slice_1443 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_522: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_521, mul_835);  add_521 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_784: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1442, 3, 64, 9223372036854775807);  slice_1442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_788: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_520, 3, 0, 64);  add_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_523: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_784, slice_scatter_788);  slice_scatter_784 = slice_scatter_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_792: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1444, 3, 64, 9223372036854775807);  slice_1444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_796: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_522, 3, 0, 64);  add_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_524: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_792, slice_scatter_796);  slice_scatter_792 = slice_scatter_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1125: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1427, [0, 2, 1, 3]);  view_1427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_251: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
    view_1434: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_251, [1, 128, 4096]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1435: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_524, [1, 128, 4096]);  add_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1436: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_523, [1, 128, 4096]);  add_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1437: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1434, [128, 4096]);  view_1434 = None
    permute_1126: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1437, [1, 0])
    mm_408: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1126, view_86);  permute_1126 = None
    permute_1127: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_408, [1, 0]);  mm_408 = None
    mm_409: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1437, permute_1128);  view_1437 = permute_1128 = None
    view_1438: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_409, [1, 128, 4096]);  mm_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_525: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1422, view_1438);  view_1422 = view_1438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1129: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1127, [1, 0]);  permute_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1439: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1435, [128, 4096]);  view_1435 = None
    permute_1130: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1439, [1, 0])
    mm_410: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1130, view_86);  permute_1130 = None
    permute_1131: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_410, [1, 0]);  mm_410 = None
    mm_411: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1439, permute_1132);  view_1439 = permute_1132 = None
    view_1440: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_411, [1, 128, 4096]);  mm_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_526: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_525, view_1440);  add_525 = view_1440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1133: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1131, [1, 0]);  permute_1131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1441: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1436, [128, 4096]);  view_1436 = None
    permute_1134: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1441, [1, 0])
    mm_412: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1134, view_86);  permute_1134 = view_86 = None
    permute_1135: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_412, [1, 0]);  mm_412 = None
    mm_413: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1441, permute_1136);  view_1441 = permute_1136 = None
    view_1442: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_413, [1, 128, 4096]);  mm_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_527: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_526, view_1442);  add_526 = view_1442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1137: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1135, [1, 0]);  permute_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_837: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_527, primals_32);  primals_32 = None
    mul_838: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_837, 4096)
    sum_213: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2], True)
    mul_839: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_837, mul_30);  mul_837 = None
    sum_214: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_839, [2], True);  mul_839 = None
    mul_840: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, sum_214);  sum_214 = None
    sub_189: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_838, sum_213);  mul_838 = sum_213 = None
    sub_190: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_189, mul_840);  sub_189 = mul_840 = None
    mul_841: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_112, sub_190);  div_112 = sub_190 = None
    mul_842: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_527, mul_30);  mul_30 = None
    sum_215: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 1]);  mul_842 = None
    sum_216: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_527, [0, 1]);  add_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_528: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_516, mul_841);  add_516 = mul_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1443: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_528, [128, 4096])
    mm_414: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1443, permute_1138);  permute_1138 = None
    permute_1139: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1443, [1, 0])
    mm_415: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1139, view_84);  view_84 = None
    permute_1140: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_415, [1, 0]);  mm_415 = None
    sum_217: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1443, [0], True)
    view_1444: "f32[4096]" = torch.ops.aten.reshape.default(sum_217, [4096]);  sum_217 = None
    permute_1141: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1140, [1, 0]);  permute_1140 = None
    view_1445: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_414, [1, 128, 16384]);  mm_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_843: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1445, mul_26);  mul_26 = None
    mul_844: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1445, add_21);  view_1445 = add_21 = None
    mul_845: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
    sub_191: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_845);  mul_845 = None
    mul_846: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_843, sub_191);  mul_843 = sub_191 = None
    mul_847: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_846, 0.7978845608028654);  mul_846 = None
    mul_848: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_847, 0.044715)
    pow_54: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_83, 2.0);  view_83 = None
    mul_849: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
    mul_850: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_529: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_847, mul_850);  mul_847 = mul_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_851: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_844, 0.5);  mul_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_530: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_529, mul_851);  add_529 = mul_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1446: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_530, [128, 16384]);  add_530 = None
    mm_416: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1446, permute_1142);  permute_1142 = None
    permute_1143: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1446, [1, 0])
    mm_417: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1143, view_58);  permute_1143 = None
    permute_1144: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_417, [1, 0]);  mm_417 = None
    sum_218: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1446, [0], True);  view_1446 = None
    view_1447: "f32[16384]" = torch.ops.aten.reshape.default(sum_218, [16384]);  sum_218 = None
    permute_1145: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1144, [1, 0]);  permute_1144 = None
    view_1448: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_416, [1, 128, 4096]);  mm_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_418: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1139, view_80);  permute_1139 = view_80 = None
    permute_1147: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_418, [1, 0]);  mm_418 = None
    mm_419: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1443, permute_1148);  view_1443 = permute_1148 = None
    view_1450: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_419, [1, 128, 4096]);  mm_419 = None
    permute_1149: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1147, [1, 0]);  permute_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1451: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1450, [1, 128, 16, 256]);  view_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1150: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1451, [0, 2, 1, 3]);  view_1451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1452: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1150, [16, 128, 256]);  permute_1150 = None
    bmm_156: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1151, view_1452);  permute_1151 = None
    bmm_157: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1452, permute_1152);  view_1452 = permute_1152 = None
    view_1453: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_156, [1, 16, 128, 256]);  bmm_156 = None
    view_1454: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_157, [1, 16, 128, 128]);  bmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_852: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1454, alias_111);  view_1454 = None
    sum_219: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [-1], True)
    mul_853: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_111, sum_219);  alias_111 = sum_219 = None
    sub_192: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_113: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_192, primals_294);  sub_192 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_61: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_144, div_113, full_default_29);  slice_144 = div_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1455: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_61, [16, 128, 128]);  where_61 = None
    bmm_158: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1153, view_1455);  permute_1153 = None
    bmm_159: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1455, permute_1154);  view_1455 = permute_1154 = None
    view_1456: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_158, [1, 16, 256, 128]);  bmm_158 = None
    view_1457: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_159, [1, 16, 128, 256]);  bmm_159 = None
    permute_1155: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1456, [0, 1, 3, 2]);  view_1456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1156: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1457, [0, 2, 1, 3]);  view_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1157: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1155, [0, 2, 1, 3]);  permute_1155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1445: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 0, 64)
    slice_1446: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 64, 256);  permute_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1447: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 0, 64)
    slice_1448: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 64, 256);  permute_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_854: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1445, view_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1458: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_854, [1, 128, 16, 32, 2]);  mul_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_100: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1458, 4, 0)
    select_101: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1458, 4, 1);  view_1458 = None
    neg_108: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_100);  select_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_800: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_108, 3, 1, 9223372036854775807, 2);  neg_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_804: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_531: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_800, slice_scatter_804);  slice_scatter_800 = slice_scatter_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_855: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1445, view_68);  slice_1445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_532: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_531, mul_855);  add_531 = mul_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_856: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1447, view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1459: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_856, [1, 128, 16, 32, 2]);  mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_102: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1459, 4, 0)
    select_103: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1459, 4, 1);  view_1459 = None
    neg_109: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_102);  select_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_808: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_109, 3, 1, 9223372036854775807, 2);  neg_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_812: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_533: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_808, slice_scatter_812);  slice_scatter_808 = slice_scatter_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_857: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1447, view_68);  slice_1447 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_534: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_533, mul_857);  add_533 = mul_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_816: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1446, 3, 64, 9223372036854775807);  slice_1446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_820: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_532, 3, 0, 64);  add_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_535: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_816, slice_scatter_820);  slice_scatter_816 = slice_scatter_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_824: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1448, 3, 64, 9223372036854775807);  slice_1448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_828: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_534, 3, 0, 64);  add_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_536: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_824, slice_scatter_828);  slice_scatter_824 = slice_scatter_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1158: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1453, [0, 2, 1, 3]);  view_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_252: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
    view_1460: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_252, [1, 128, 4096]);  clone_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1461: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_536, [1, 128, 4096]);  add_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1462: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_535, [1, 128, 4096]);  add_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1463: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1460, [128, 4096]);  view_1460 = None
    permute_1159: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1463, [1, 0])
    mm_420: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1159, view_58);  permute_1159 = None
    permute_1160: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_420, [1, 0]);  mm_420 = None
    mm_421: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1463, permute_1161);  view_1463 = permute_1161 = None
    view_1464: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_421, [1, 128, 4096]);  mm_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_537: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1448, view_1464);  view_1448 = view_1464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1162: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1160, [1, 0]);  permute_1160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1465: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1461, [128, 4096]);  view_1461 = None
    permute_1163: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1465, [1, 0])
    mm_422: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1163, view_58);  permute_1163 = None
    permute_1164: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_422, [1, 0]);  mm_422 = None
    mm_423: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1465, permute_1165);  view_1465 = permute_1165 = None
    view_1466: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_423, [1, 128, 4096]);  mm_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_538: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_537, view_1466);  add_537 = view_1466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1166: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1164, [1, 0]);  permute_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1467: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1462, [128, 4096]);  view_1462 = None
    permute_1167: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1467, [1, 0])
    mm_424: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1167, view_58);  permute_1167 = view_58 = None
    permute_1168: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_424, [1, 0]);  mm_424 = None
    mm_425: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1467, permute_1169);  view_1467 = permute_1169 = None
    view_1468: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_425, [1, 128, 4096]);  mm_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_539: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_538, view_1468);  add_538 = view_1468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1170: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1168, [1, 0]);  permute_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_859: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_539, primals_22);  primals_22 = None
    mul_860: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_859, 4096)
    sum_220: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_859, mul_20);  mul_859 = None
    sum_221: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, sum_221);  sum_221 = None
    sub_194: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_860, sum_220);  mul_860 = sum_220 = None
    sub_195: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_194, mul_862);  sub_194 = mul_862 = None
    mul_863: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_114, sub_195);  div_114 = sub_195 = None
    mul_864: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_539, mul_20);  mul_20 = None
    sum_222: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
    sum_223: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_539, [0, 1]);  add_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_540: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_528, mul_863);  add_528 = mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1469: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_540, [128, 4096])
    mm_426: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1469, permute_1171);  permute_1171 = None
    permute_1172: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1469, [1, 0])
    mm_427: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1172, view_56);  view_56 = None
    permute_1173: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_427, [1, 0]);  mm_427 = None
    sum_224: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1469, [0], True)
    view_1470: "f32[4096]" = torch.ops.aten.reshape.default(sum_224, [4096]);  sum_224 = None
    permute_1174: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1173, [1, 0]);  permute_1173 = None
    view_1471: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_426, [1, 128, 16384]);  mm_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_865: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1471, mul_16);  mul_16 = None
    mul_866: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1471, add_13);  view_1471 = add_13 = None
    mul_867: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
    sub_196: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_867);  mul_867 = None
    mul_868: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_865, sub_196);  mul_865 = sub_196 = None
    mul_869: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_868, 0.7978845608028654);  mul_868 = None
    mul_870: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_869, 0.044715)
    pow_55: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_55, 2.0);  view_55 = None
    mul_871: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
    mul_872: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_541: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_869, mul_872);  mul_869 = mul_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_873: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_866, 0.5);  mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_542: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_541, mul_873);  add_541 = mul_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1472: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_542, [128, 16384]);  add_542 = None
    mm_428: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1472, permute_1175);  permute_1175 = None
    permute_1176: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1472, [1, 0])
    mm_429: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1176, view_30);  permute_1176 = None
    permute_1177: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_429, [1, 0]);  mm_429 = None
    sum_225: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1472, [0], True);  view_1472 = None
    view_1473: "f32[16384]" = torch.ops.aten.reshape.default(sum_225, [16384]);  sum_225 = None
    permute_1178: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1177, [1, 0]);  permute_1177 = None
    view_1474: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_428, [1, 128, 4096]);  mm_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_430: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1172, view_52);  permute_1172 = view_52 = None
    permute_1180: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_430, [1, 0]);  mm_430 = None
    mm_431: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1469, permute_1181);  view_1469 = permute_1181 = None
    view_1476: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_431, [1, 128, 4096]);  mm_431 = None
    permute_1182: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1180, [1, 0]);  permute_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1477: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1476, [1, 128, 16, 256]);  view_1476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1183: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1477, [0, 2, 1, 3]);  view_1477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1478: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1183, [16, 128, 256]);  permute_1183 = None
    bmm_160: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1184, view_1478);  permute_1184 = None
    bmm_161: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1478, permute_1185);  view_1478 = permute_1185 = None
    view_1479: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_160, [1, 16, 128, 256]);  bmm_160 = None
    view_1480: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_161, [1, 16, 128, 128]);  bmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_874: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1480, alias_113);  view_1480 = None
    sum_226: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_874, [-1], True)
    mul_875: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_113, sum_226);  alias_113 = sum_226 = None
    sub_197: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_874, mul_875);  mul_874 = mul_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_115: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_197, primals_291);  sub_197 = primals_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_62: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, div_115, full_default_29);  slice_96 = div_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1481: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_62, [16, 128, 128]);  where_62 = None
    bmm_162: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1186, view_1481);  permute_1186 = None
    bmm_163: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1481, permute_1187);  view_1481 = permute_1187 = None
    view_1482: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_162, [1, 16, 256, 128]);  bmm_162 = None
    view_1483: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_163, [1, 16, 128, 256]);  bmm_163 = None
    permute_1188: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1482, [0, 1, 3, 2]);  view_1482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1189: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1483, [0, 2, 1, 3]);  view_1483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1190: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1188, [0, 2, 1, 3]);  permute_1188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1449: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 0, 64)
    slice_1450: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 64, 256);  permute_1189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1451: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 0, 64)
    slice_1452: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 64, 256);  permute_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_876: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1449, view_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1484: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_876, [1, 128, 16, 32, 2]);  mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_104: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1484, 4, 0)
    select_105: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1484, 4, 1);  view_1484 = None
    neg_110: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_104);  select_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_832: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_110, 3, 1, 9223372036854775807, 2);  neg_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_836: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_543: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_832, slice_scatter_836);  slice_scatter_832 = slice_scatter_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_877: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1449, view_40);  slice_1449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_544: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_543, mul_877);  add_543 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_878: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1451, view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1485: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_878, [1, 128, 16, 32, 2]);  mul_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_106: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1485, 4, 0)
    select_107: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1485, 4, 1);  view_1485 = None
    neg_111: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_106);  select_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_840: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_111, 3, 1, 9223372036854775807, 2);  neg_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_844: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_545: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_840, slice_scatter_844);  slice_scatter_840 = slice_scatter_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_879: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1451, view_40);  slice_1451 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_546: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_545, mul_879);  add_545 = mul_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_848: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1450, 3, 64, 9223372036854775807);  slice_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_852: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_544, 3, 0, 64);  add_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_547: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_848, slice_scatter_852);  slice_scatter_848 = slice_scatter_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_856: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1452, 3, 64, 9223372036854775807);  slice_1452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_860: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_546, 3, 0, 64);  add_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_548: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_856, slice_scatter_860);  slice_scatter_856 = slice_scatter_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1191: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1479, [0, 2, 1, 3]);  view_1479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_253: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
    view_1486: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_253, [1, 128, 4096]);  clone_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1487: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_548, [1, 128, 4096]);  add_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1488: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_547, [1, 128, 4096]);  add_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1489: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1486, [128, 4096]);  view_1486 = None
    permute_1192: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1489, [1, 0])
    mm_432: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1192, view_30);  permute_1192 = None
    permute_1193: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_432, [1, 0]);  mm_432 = None
    mm_433: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1489, permute_1194);  view_1489 = permute_1194 = None
    view_1490: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_433, [1, 128, 4096]);  mm_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_549: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1474, view_1490);  view_1474 = view_1490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1195: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1193, [1, 0]);  permute_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1491: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1487, [128, 4096]);  view_1487 = None
    permute_1196: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1491, [1, 0])
    mm_434: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1196, view_30);  permute_1196 = None
    permute_1197: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_434, [1, 0]);  mm_434 = None
    mm_435: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1491, permute_1198);  view_1491 = permute_1198 = None
    view_1492: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_435, [1, 128, 4096]);  mm_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_550: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_549, view_1492);  add_549 = view_1492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1199: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1197, [1, 0]);  permute_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1493: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1488, [128, 4096]);  view_1488 = None
    permute_1200: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1493, [1, 0])
    mm_436: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1200, view_30);  permute_1200 = view_30 = None
    permute_1201: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_436, [1, 0]);  mm_436 = None
    mm_437: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1493, permute_1202);  view_1493 = permute_1202 = None
    view_1494: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_437, [1, 128, 4096]);  mm_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_551: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_550, view_1494);  add_550 = view_1494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1203: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1201, [1, 0]);  permute_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_881: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_551, primals_12);  primals_12 = None
    mul_882: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_881, 4096)
    sum_227: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_881, [2], True)
    mul_883: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_881, mul_10);  mul_881 = None
    sum_228: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_883, [2], True);  mul_883 = None
    mul_884: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_228);  sum_228 = None
    sub_199: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_882, sum_227);  mul_882 = sum_227 = None
    sub_200: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_199, mul_884);  sub_199 = mul_884 = None
    mul_885: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_116, sub_200);  div_116 = sub_200 = None
    mul_886: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_551, mul_10);  mul_10 = None
    sum_229: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 1]);  mul_886 = None
    sum_230: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_551, [0, 1]);  add_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_552: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_540, mul_885);  add_540 = mul_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_1495: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_552, [128, 4096])
    mm_438: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1495, permute_1204);  permute_1204 = None
    permute_1205: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1495, [1, 0])
    mm_439: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1205, view_28);  view_28 = None
    permute_1206: "f32[16384, 4096]" = torch.ops.aten.permute.default(mm_439, [1, 0]);  mm_439 = None
    sum_231: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1495, [0], True)
    view_1496: "f32[4096]" = torch.ops.aten.reshape.default(sum_231, [4096]);  sum_231 = None
    permute_1207: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_1206, [1, 0]);  permute_1206 = None
    view_1497: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_438, [1, 128, 16384]);  mm_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_887: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1497, mul_6);  mul_6 = None
    mul_888: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1497, add_5);  view_1497 = add_5 = None
    mul_889: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
    sub_201: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_889);  mul_889 = None
    mul_890: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_887, sub_201);  mul_887 = sub_201 = None
    mul_891: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_890, 0.7978845608028654);  mul_890 = None
    mul_892: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_891, 0.044715)
    pow_56: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 2.0);  view_27 = None
    mul_893: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
    mul_894: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_892, mul_893);  mul_892 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_553: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_891, mul_894);  mul_891 = mul_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_895: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_888, 0.5);  mul_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_554: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_553, mul_895);  add_553 = mul_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_1498: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_554, [128, 16384]);  add_554 = None
    mm_440: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1498, permute_1208);  permute_1208 = None
    permute_1209: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1498, [1, 0])
    mm_441: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1209, view_2);  permute_1209 = None
    permute_1210: "f32[4096, 16384]" = torch.ops.aten.permute.default(mm_441, [1, 0]);  mm_441 = None
    sum_232: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1498, [0], True);  view_1498 = None
    view_1499: "f32[16384]" = torch.ops.aten.reshape.default(sum_232, [16384]);  sum_232 = None
    permute_1211: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_1210, [1, 0]);  permute_1210 = None
    view_1500: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_440, [1, 128, 4096]);  mm_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    mm_442: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1205, view_24);  permute_1205 = view_24 = None
    permute_1213: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_442, [1, 0]);  mm_442 = None
    mm_443: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1495, permute_1214);  view_1495 = permute_1214 = None
    view_1502: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_443, [1, 128, 4096]);  mm_443 = None
    permute_1215: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1213, [1, 0]);  permute_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_1503: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1502, [1, 128, 16, 256]);  view_1502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1216: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1503, [0, 2, 1, 3]);  view_1503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    view_1504: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(permute_1216, [16, 128, 256]);  permute_1216 = None
    bmm_164: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(permute_1217, view_1504);  permute_1217 = None
    bmm_165: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1504, permute_1218);  view_1504 = permute_1218 = None
    view_1505: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_164, [1, 16, 128, 256]);  bmm_164 = None
    view_1506: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_165, [1, 16, 128, 128]);  bmm_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    mul_896: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1506, alias_115);  view_1506 = None
    sum_233: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [-1], True)
    mul_897: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_115, sum_233);  alias_115 = sum_233 = None
    sub_202: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_117: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(sub_202, primals_288);  sub_202 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_63: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, div_117, full_default_29);  slice_48 = div_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1507: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(where_63, [16, 128, 128]);  where_63 = None
    bmm_166: "f32[16, 256, 128]" = torch.ops.aten.bmm.default(permute_1219, view_1507);  permute_1219 = None
    bmm_167: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_1507, permute_1220);  view_1507 = permute_1220 = None
    view_1508: "f32[1, 16, 256, 128]" = torch.ops.aten.reshape.default(bmm_166, [1, 16, 256, 128]);  bmm_166 = None
    view_1509: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_167, [1, 16, 128, 256]);  bmm_167 = None
    permute_1221: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1508, [0, 1, 3, 2]);  view_1508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_1222: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1509, [0, 2, 1, 3]);  view_1509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_1223: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(permute_1221, [0, 2, 1, 3]);  permute_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    slice_1453: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 0, 64)
    slice_1454: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 64, 256);  permute_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    slice_1455: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 0, 64)
    slice_1456: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 64, 256);  permute_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_898: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1453, view_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1510: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_898, [1, 128, 16, 32, 2]);  mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_108: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1510, 4, 0)
    select_109: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1510, 4, 1);  view_1510 = None
    neg_112: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_108);  select_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_864: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_112, 3, 1, 9223372036854775807, 2);  neg_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_868: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_555: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_864, slice_scatter_868);  slice_scatter_864 = slice_scatter_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_899: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1453, view_12);  slice_1453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_556: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_555, mul_899);  add_555 = mul_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_900: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1455, view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_1511: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_900, [1, 128, 16, 32, 2]);  mul_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    select_110: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1511, 4, 0)
    select_111: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1511, 4, 1);  view_1511 = None
    neg_113: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_110);  select_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_scatter_872: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, neg_113, 3, 1, 9223372036854775807, 2);  neg_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_scatter_876: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_39, select_111, 3, 0, 9223372036854775807, 2);  full_default_39 = select_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    add_557: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_872, slice_scatter_876);  slice_scatter_872 = slice_scatter_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_901: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1455, view_12);  slice_1455 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    add_558: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_557, mul_901);  add_557 = mul_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_scatter_880: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1454, 3, 64, 9223372036854775807);  slice_1454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_scatter_884: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_556, 3, 0, 64);  add_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    add_559: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_880, slice_scatter_884);  slice_scatter_880 = slice_scatter_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_scatter_888: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, slice_1456, 3, 64, 9223372036854775807);  slice_1456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_scatter_892: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_55, add_558, 3, 0, 64);  full_default_55 = add_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    add_560: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_888, slice_scatter_892);  slice_scatter_888 = slice_scatter_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1224: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_1505, [0, 2, 1, 3]);  view_1505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    clone_254: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
    view_1512: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_254, [1, 128, 4096]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1513: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_560, [1, 128, 4096]);  add_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_1514: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_559, [1, 128, 4096]);  add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_1515: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1512, [128, 4096]);  view_1512 = None
    permute_1225: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1515, [1, 0])
    mm_444: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1225, view_2);  permute_1225 = None
    permute_1226: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_444, [1, 0]);  mm_444 = None
    mm_445: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1515, permute_1227);  view_1515 = permute_1227 = None
    view_1516: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_445, [1, 128, 4096]);  mm_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    add_561: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1500, view_1516);  view_1500 = view_1516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    permute_1228: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1226, [1, 0]);  permute_1226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_1517: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1513, [128, 4096]);  view_1513 = None
    permute_1229: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1517, [1, 0])
    mm_446: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1229, view_2);  permute_1229 = None
    permute_1230: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_446, [1, 0]);  mm_446 = None
    mm_447: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1517, permute_1231);  view_1517 = permute_1231 = None
    view_1518: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_447, [1, 128, 4096]);  mm_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    add_562: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_561, view_1518);  add_561 = view_1518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    permute_1232: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1230, [1, 0]);  permute_1230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_1519: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1514, [128, 4096]);  view_1514 = None
    permute_1233: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1519, [1, 0])
    mm_448: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1233, view_2);  permute_1233 = view_2 = None
    permute_1234: "f32[4096, 4096]" = torch.ops.aten.permute.default(mm_448, [1, 0]);  mm_448 = None
    mm_449: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1519, permute_1235);  view_1519 = permute_1235 = None
    view_1520: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_449, [1, 128, 4096]);  mm_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    add_563: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_562, view_1520);  add_562 = view_1520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    permute_1236: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1234, [1, 0]);  permute_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    mul_903: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_563, primals_2);  primals_2 = None
    mul_904: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_903, 4096)
    sum_234: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_903, [2], True)
    mul_905: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_903, mul);  mul_903 = None
    sum_235: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_905, [2], True);  mul_905 = None
    mul_906: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, sum_235);  sum_235 = None
    sub_204: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_904, sum_234);  mul_904 = sum_234 = None
    sub_205: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_204, mul_906);  sub_204 = mul_906 = None
    div_118: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
    mul_907: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_118, sub_205);  div_118 = sub_205 = None
    mul_908: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_563, mul);  mul = None
    sum_236: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_908, [0, 1]);  mul_908 = None
    sum_237: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_563, [0, 1]);  add_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    add_564: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_552, mul_907);  add_552 = mul_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:635, code: inputs_embeds = self.wte(input_ids)
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_371: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_64: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_371, full_default_29, add_564);  unsqueeze_371 = full_default_29 = add_564 = None
    full_default_963: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[50400, 4096]" = torch.ops.prims._unsafe_index_put_.default(full_default_963, [view], where_64, True);  full_default_963 = view = where_64 = None
    return [_unsafe_index_put, sum_236, sum_237, permute_1236, permute_1232, permute_1228, permute_1215, permute_1211, view_1499, permute_1207, view_1496, sum_229, sum_230, permute_1203, permute_1199, permute_1195, permute_1182, permute_1178, view_1473, permute_1174, view_1470, sum_222, sum_223, permute_1170, permute_1166, permute_1162, permute_1149, permute_1145, view_1447, permute_1141, view_1444, sum_215, sum_216, permute_1137, permute_1133, permute_1129, permute_1116, permute_1112, view_1421, permute_1108, view_1418, sum_208, sum_209, permute_1104, permute_1100, permute_1096, permute_1083, permute_1079, view_1395, permute_1075, view_1392, sum_201, sum_202, permute_1071, permute_1067, permute_1063, permute_1050, permute_1046, view_1369, permute_1042, view_1366, sum_194, sum_195, permute_1038, permute_1034, permute_1030, permute_1017, permute_1013, view_1343, permute_1009, view_1340, sum_187, sum_188, permute_1005, permute_1001, permute_997, permute_984, permute_980, view_1317, permute_976, view_1314, sum_180, sum_181, permute_972, permute_968, permute_964, permute_951, permute_947, view_1291, permute_943, view_1288, sum_173, sum_174, permute_939, permute_935, permute_931, permute_918, permute_914, view_1265, permute_910, view_1262, sum_166, sum_167, permute_906, permute_902, permute_898, permute_885, permute_881, view_1239, permute_877, view_1236, sum_159, sum_160, permute_873, permute_869, permute_865, permute_852, permute_848, view_1213, permute_844, view_1210, sum_152, sum_153, permute_840, permute_836, permute_832, permute_819, permute_815, view_1187, permute_811, view_1184, sum_145, sum_146, permute_807, permute_803, permute_799, permute_786, permute_782, view_1161, permute_778, view_1158, sum_138, sum_139, permute_774, permute_770, permute_766, permute_753, permute_749, view_1135, permute_745, view_1132, sum_131, sum_132, permute_741, permute_737, permute_733, permute_720, permute_716, view_1109, permute_712, view_1106, sum_124, sum_125, permute_708, permute_704, permute_700, permute_687, permute_683, view_1083, permute_679, view_1080, sum_117, sum_118, permute_675, permute_671, permute_667, permute_654, permute_650, view_1057, permute_646, view_1054, sum_110, sum_111, permute_642, permute_638, permute_634, permute_621, permute_617, view_1031, permute_613, view_1028, sum_103, sum_104, permute_609, permute_605, permute_601, permute_588, permute_584, view_1005, permute_580, view_1002, sum_96, sum_97, permute_576, permute_572, permute_568, permute_555, permute_551, view_979, permute_547, view_976, sum_89, sum_90, permute_543, permute_539, permute_535, permute_522, permute_518, view_953, permute_514, view_950, sum_82, sum_83, permute_510, permute_506, permute_502, permute_489, permute_485, view_927, permute_481, view_924, sum_75, sum_76, permute_477, permute_473, permute_469, permute_456, permute_452, view_901, permute_448, view_898, sum_68, sum_69, permute_444, permute_440, permute_436, permute_423, permute_419, view_875, permute_415, view_872, sum_61, sum_62, permute_411, permute_407, permute_403, permute_390, permute_386, view_849, permute_382, view_846, sum_54, sum_55, permute_378, permute_374, permute_370, permute_357, permute_353, view_823, permute_349, view_820, sum_47, sum_48, permute_345, permute_341, permute_337, permute_324, permute_320, view_797, permute_316, view_794, sum_40, sum_41, permute_312, view_790, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    