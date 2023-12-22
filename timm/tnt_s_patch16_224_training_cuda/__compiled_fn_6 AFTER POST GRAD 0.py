from __future__ import annotations



def forward(self, primals_4: "f32[24, 3, 7, 7]", primals_6: "f32[384]", primals_10: "f32[384]", primals_12: "f32[24]", primals_18: "f32[24]", primals_24: "f32[24]", primals_28: "f32[384]", primals_34: "f32[384]", primals_40: "f32[24]", primals_46: "f32[24]", primals_52: "f32[24]", primals_56: "f32[384]", primals_62: "f32[384]", primals_68: "f32[24]", primals_74: "f32[24]", primals_80: "f32[24]", primals_84: "f32[384]", primals_90: "f32[384]", primals_96: "f32[24]", primals_102: "f32[24]", primals_108: "f32[24]", primals_112: "f32[384]", primals_118: "f32[384]", primals_124: "f32[24]", primals_130: "f32[24]", primals_136: "f32[24]", primals_140: "f32[384]", primals_146: "f32[384]", primals_152: "f32[24]", primals_158: "f32[24]", primals_164: "f32[24]", primals_168: "f32[384]", primals_174: "f32[384]", primals_180: "f32[24]", primals_186: "f32[24]", primals_192: "f32[24]", primals_196: "f32[384]", primals_202: "f32[384]", primals_208: "f32[24]", primals_214: "f32[24]", primals_220: "f32[24]", primals_224: "f32[384]", primals_230: "f32[384]", primals_236: "f32[24]", primals_242: "f32[24]", primals_248: "f32[24]", primals_252: "f32[384]", primals_258: "f32[384]", primals_264: "f32[24]", primals_270: "f32[24]", primals_276: "f32[24]", primals_280: "f32[384]", primals_286: "f32[384]", primals_292: "f32[24]", primals_298: "f32[24]", primals_304: "f32[24]", primals_308: "f32[384]", primals_314: "f32[384]", primals_320: "f32[24]", primals_326: "f32[24]", primals_332: "f32[24]", primals_336: "f32[384]", primals_342: "f32[384]", primals_348: "f32[384]", primals_352: "f32[8, 3, 224, 224]", add: "i64[4, 14]", unsqueeze_5: "i64[4, 14, 1, 1]", clone_2: "f32[1568, 16, 24]", getitem_1: "f32[8, 196, 1]", rsqrt: "f32[8, 196, 1]", view_4: "f32[1568, 384]", addmm: "f32[1568, 384]", getitem_3: "f32[8, 196, 1]", rsqrt_1: "f32[8, 196, 1]", getitem_5: "f32[1568, 16, 1]", rsqrt_2: "f32[1568, 16, 1]", view_6: "f32[25088, 24]", view_19: "f32[25088, 24]", mul_7: "f32[1568, 16, 24]", view_21: "f32[25088, 24]", addmm_2: "f32[25088, 96]", view_23: "f32[25088, 96]", mul_12: "f32[1568, 16, 24]", view_26: "f32[1568, 384]", cat_1: "f32[8, 197, 384]", getitem_13: "f32[8, 197, 1]", rsqrt_5: "f32[8, 197, 1]", view_28: "f32[1576, 384]", view_41: "f32[1576, 384]", mul_17: "f32[8, 197, 384]", view_43: "f32[1576, 384]", addmm_6: "f32[1576, 1536]", view_45: "f32[1576, 1536]", view_47: "f32[25088, 24]", view_60: "f32[25088, 24]", mul_25: "f32[1568, 16, 24]", view_62: "f32[25088, 24]", addmm_9: "f32[25088, 96]", view_64: "f32[25088, 96]", mul_30: "f32[1568, 16, 24]", view_67: "f32[1568, 384]", cat_2: "f32[8, 197, 384]", getitem_27: "f32[8, 197, 1]", rsqrt_10: "f32[8, 197, 1]", view_69: "f32[1576, 384]", view_82: "f32[1576, 384]", mul_35: "f32[8, 197, 384]", view_84: "f32[1576, 384]", addmm_13: "f32[1576, 1536]", view_86: "f32[1576, 1536]", view_88: "f32[25088, 24]", view_101: "f32[25088, 24]", mul_43: "f32[1568, 16, 24]", view_103: "f32[25088, 24]", addmm_16: "f32[25088, 96]", view_105: "f32[25088, 96]", mul_48: "f32[1568, 16, 24]", view_108: "f32[1568, 384]", cat_3: "f32[8, 197, 384]", getitem_41: "f32[8, 197, 1]", rsqrt_15: "f32[8, 197, 1]", view_110: "f32[1576, 384]", view_123: "f32[1576, 384]", mul_53: "f32[8, 197, 384]", view_125: "f32[1576, 384]", addmm_20: "f32[1576, 1536]", view_127: "f32[1576, 1536]", view_129: "f32[25088, 24]", view_142: "f32[25088, 24]", mul_61: "f32[1568, 16, 24]", view_144: "f32[25088, 24]", addmm_23: "f32[25088, 96]", view_146: "f32[25088, 96]", mul_66: "f32[1568, 16, 24]", view_149: "f32[1568, 384]", cat_4: "f32[8, 197, 384]", getitem_55: "f32[8, 197, 1]", rsqrt_20: "f32[8, 197, 1]", view_151: "f32[1576, 384]", view_164: "f32[1576, 384]", mul_71: "f32[8, 197, 384]", view_166: "f32[1576, 384]", addmm_27: "f32[1576, 1536]", view_168: "f32[1576, 1536]", view_170: "f32[25088, 24]", view_183: "f32[25088, 24]", mul_79: "f32[1568, 16, 24]", view_185: "f32[25088, 24]", addmm_30: "f32[25088, 96]", view_187: "f32[25088, 96]", mul_84: "f32[1568, 16, 24]", view_190: "f32[1568, 384]", cat_5: "f32[8, 197, 384]", getitem_69: "f32[8, 197, 1]", rsqrt_25: "f32[8, 197, 1]", view_192: "f32[1576, 384]", view_205: "f32[1576, 384]", mul_89: "f32[8, 197, 384]", view_207: "f32[1576, 384]", addmm_34: "f32[1576, 1536]", view_209: "f32[1576, 1536]", view_211: "f32[25088, 24]", view_224: "f32[25088, 24]", mul_97: "f32[1568, 16, 24]", view_226: "f32[25088, 24]", addmm_37: "f32[25088, 96]", view_228: "f32[25088, 96]", mul_102: "f32[1568, 16, 24]", view_231: "f32[1568, 384]", cat_6: "f32[8, 197, 384]", getitem_83: "f32[8, 197, 1]", rsqrt_30: "f32[8, 197, 1]", view_233: "f32[1576, 384]", view_246: "f32[1576, 384]", mul_107: "f32[8, 197, 384]", view_248: "f32[1576, 384]", addmm_41: "f32[1576, 1536]", view_250: "f32[1576, 1536]", view_252: "f32[25088, 24]", view_265: "f32[25088, 24]", mul_115: "f32[1568, 16, 24]", view_267: "f32[25088, 24]", addmm_44: "f32[25088, 96]", view_269: "f32[25088, 96]", mul_120: "f32[1568, 16, 24]", view_272: "f32[1568, 384]", cat_7: "f32[8, 197, 384]", getitem_97: "f32[8, 197, 1]", rsqrt_35: "f32[8, 197, 1]", view_274: "f32[1576, 384]", view_287: "f32[1576, 384]", mul_125: "f32[8, 197, 384]", view_289: "f32[1576, 384]", addmm_48: "f32[1576, 1536]", view_291: "f32[1576, 1536]", view_293: "f32[25088, 24]", view_306: "f32[25088, 24]", mul_133: "f32[1568, 16, 24]", view_308: "f32[25088, 24]", addmm_51: "f32[25088, 96]", view_310: "f32[25088, 96]", mul_138: "f32[1568, 16, 24]", view_313: "f32[1568, 384]", cat_8: "f32[8, 197, 384]", getitem_111: "f32[8, 197, 1]", rsqrt_40: "f32[8, 197, 1]", view_315: "f32[1576, 384]", view_328: "f32[1576, 384]", mul_143: "f32[8, 197, 384]", view_330: "f32[1576, 384]", addmm_55: "f32[1576, 1536]", view_332: "f32[1576, 1536]", view_334: "f32[25088, 24]", view_347: "f32[25088, 24]", mul_151: "f32[1568, 16, 24]", view_349: "f32[25088, 24]", addmm_58: "f32[25088, 96]", view_351: "f32[25088, 96]", mul_156: "f32[1568, 16, 24]", view_354: "f32[1568, 384]", cat_9: "f32[8, 197, 384]", getitem_125: "f32[8, 197, 1]", rsqrt_45: "f32[8, 197, 1]", view_356: "f32[1576, 384]", view_369: "f32[1576, 384]", mul_161: "f32[8, 197, 384]", view_371: "f32[1576, 384]", addmm_62: "f32[1576, 1536]", view_373: "f32[1576, 1536]", view_375: "f32[25088, 24]", view_388: "f32[25088, 24]", mul_169: "f32[1568, 16, 24]", view_390: "f32[25088, 24]", addmm_65: "f32[25088, 96]", view_392: "f32[25088, 96]", mul_174: "f32[1568, 16, 24]", view_395: "f32[1568, 384]", cat_10: "f32[8, 197, 384]", getitem_139: "f32[8, 197, 1]", rsqrt_50: "f32[8, 197, 1]", view_397: "f32[1576, 384]", view_410: "f32[1576, 384]", mul_179: "f32[8, 197, 384]", view_412: "f32[1576, 384]", addmm_69: "f32[1576, 1536]", view_414: "f32[1576, 1536]", view_416: "f32[25088, 24]", view_429: "f32[25088, 24]", mul_187: "f32[1568, 16, 24]", view_431: "f32[25088, 24]", addmm_72: "f32[25088, 96]", view_433: "f32[25088, 96]", mul_192: "f32[1568, 16, 24]", view_436: "f32[1568, 384]", cat_11: "f32[8, 197, 384]", getitem_153: "f32[8, 197, 1]", rsqrt_55: "f32[8, 197, 1]", view_438: "f32[1576, 384]", view_451: "f32[1576, 384]", mul_197: "f32[8, 197, 384]", view_453: "f32[1576, 384]", addmm_76: "f32[1576, 1536]", view_455: "f32[1576, 1536]", view_457: "f32[25088, 24]", view_470: "f32[25088, 24]", mul_205: "f32[1568, 16, 24]", view_472: "f32[25088, 24]", addmm_79: "f32[25088, 96]", view_474: "f32[25088, 96]", mul_210: "f32[1568, 16, 24]", view_477: "f32[1568, 384]", cat_12: "f32[8, 197, 384]", getitem_167: "f32[8, 197, 1]", rsqrt_60: "f32[8, 197, 1]", view_479: "f32[1576, 384]", view_492: "f32[1576, 384]", mul_215: "f32[8, 197, 384]", view_494: "f32[1576, 384]", addmm_83: "f32[1576, 1536]", view_496: "f32[1576, 1536]", mul_220: "f32[8, 197, 384]", clone_184: "f32[8, 384]", permute_233: "f32[1000, 384]", div_24: "f32[8, 197, 1]", permute_237: "f32[384, 1536]", permute_241: "f32[1536, 384]", div_25: "f32[8, 197, 1]", permute_245: "f32[384, 384]", permute_250: "f32[48, 197, 197]", permute_251: "f32[48, 64, 197]", alias_24: "f32[8, 6, 197, 197]", permute_252: "f32[48, 64, 197]", permute_253: "f32[48, 197, 64]", permute_258: "f32[384, 384]", permute_263: "f32[768, 384]", permute_265: "f32[384, 384]", div_27: "f32[1568, 16, 1]", permute_269: "f32[24, 96]", permute_273: "f32[96, 24]", div_28: "f32[1568, 16, 1]", permute_277: "f32[24, 24]", permute_282: "f32[6272, 16, 16]", permute_283: "f32[6272, 6, 16]", alias_25: "f32[1568, 4, 16, 16]", permute_284: "f32[6272, 6, 16]", permute_285: "f32[6272, 16, 6]", permute_290: "f32[24, 24]", permute_295: "f32[48, 24]", div_29: "f32[1568, 16, 1]", permute_297: "f32[384, 1536]", permute_301: "f32[1536, 384]", div_30: "f32[8, 197, 1]", permute_305: "f32[384, 384]", permute_310: "f32[48, 197, 197]", permute_311: "f32[48, 64, 197]", alias_26: "f32[8, 6, 197, 197]", permute_312: "f32[48, 64, 197]", permute_313: "f32[48, 197, 64]", permute_318: "f32[384, 384]", permute_323: "f32[768, 384]", permute_325: "f32[384, 384]", permute_329: "f32[24, 96]", permute_333: "f32[96, 24]", div_33: "f32[1568, 16, 1]", permute_337: "f32[24, 24]", permute_342: "f32[6272, 16, 16]", permute_343: "f32[6272, 6, 16]", alias_27: "f32[1568, 4, 16, 16]", permute_344: "f32[6272, 6, 16]", permute_345: "f32[6272, 16, 6]", permute_350: "f32[24, 24]", permute_355: "f32[48, 24]", div_34: "f32[1568, 16, 1]", permute_357: "f32[384, 1536]", permute_361: "f32[1536, 384]", div_35: "f32[8, 197, 1]", permute_365: "f32[384, 384]", permute_370: "f32[48, 197, 197]", permute_371: "f32[48, 64, 197]", alias_28: "f32[8, 6, 197, 197]", permute_372: "f32[48, 64, 197]", permute_373: "f32[48, 197, 64]", permute_378: "f32[384, 384]", permute_383: "f32[768, 384]", permute_385: "f32[384, 384]", permute_389: "f32[24, 96]", permute_393: "f32[96, 24]", div_38: "f32[1568, 16, 1]", permute_397: "f32[24, 24]", permute_402: "f32[6272, 16, 16]", permute_403: "f32[6272, 6, 16]", alias_29: "f32[1568, 4, 16, 16]", permute_404: "f32[6272, 6, 16]", permute_405: "f32[6272, 16, 6]", permute_410: "f32[24, 24]", permute_415: "f32[48, 24]", div_39: "f32[1568, 16, 1]", permute_417: "f32[384, 1536]", permute_421: "f32[1536, 384]", div_40: "f32[8, 197, 1]", permute_425: "f32[384, 384]", permute_430: "f32[48, 197, 197]", permute_431: "f32[48, 64, 197]", alias_30: "f32[8, 6, 197, 197]", permute_432: "f32[48, 64, 197]", permute_433: "f32[48, 197, 64]", permute_438: "f32[384, 384]", permute_443: "f32[768, 384]", permute_445: "f32[384, 384]", permute_449: "f32[24, 96]", permute_453: "f32[96, 24]", div_43: "f32[1568, 16, 1]", permute_457: "f32[24, 24]", permute_462: "f32[6272, 16, 16]", permute_463: "f32[6272, 6, 16]", alias_31: "f32[1568, 4, 16, 16]", permute_464: "f32[6272, 6, 16]", permute_465: "f32[6272, 16, 6]", permute_470: "f32[24, 24]", permute_475: "f32[48, 24]", div_44: "f32[1568, 16, 1]", permute_477: "f32[384, 1536]", permute_481: "f32[1536, 384]", div_45: "f32[8, 197, 1]", permute_485: "f32[384, 384]", permute_490: "f32[48, 197, 197]", permute_491: "f32[48, 64, 197]", alias_32: "f32[8, 6, 197, 197]", permute_492: "f32[48, 64, 197]", permute_493: "f32[48, 197, 64]", permute_498: "f32[384, 384]", permute_503: "f32[768, 384]", permute_505: "f32[384, 384]", permute_509: "f32[24, 96]", permute_513: "f32[96, 24]", div_48: "f32[1568, 16, 1]", permute_517: "f32[24, 24]", permute_522: "f32[6272, 16, 16]", permute_523: "f32[6272, 6, 16]", alias_33: "f32[1568, 4, 16, 16]", permute_524: "f32[6272, 6, 16]", permute_525: "f32[6272, 16, 6]", permute_530: "f32[24, 24]", permute_535: "f32[48, 24]", div_49: "f32[1568, 16, 1]", permute_537: "f32[384, 1536]", permute_541: "f32[1536, 384]", div_50: "f32[8, 197, 1]", permute_545: "f32[384, 384]", permute_550: "f32[48, 197, 197]", permute_551: "f32[48, 64, 197]", alias_34: "f32[8, 6, 197, 197]", permute_552: "f32[48, 64, 197]", permute_553: "f32[48, 197, 64]", permute_558: "f32[384, 384]", permute_563: "f32[768, 384]", permute_565: "f32[384, 384]", permute_569: "f32[24, 96]", permute_573: "f32[96, 24]", div_53: "f32[1568, 16, 1]", permute_577: "f32[24, 24]", permute_582: "f32[6272, 16, 16]", permute_583: "f32[6272, 6, 16]", alias_35: "f32[1568, 4, 16, 16]", permute_584: "f32[6272, 6, 16]", permute_585: "f32[6272, 16, 6]", permute_590: "f32[24, 24]", permute_595: "f32[48, 24]", div_54: "f32[1568, 16, 1]", permute_597: "f32[384, 1536]", permute_601: "f32[1536, 384]", div_55: "f32[8, 197, 1]", permute_605: "f32[384, 384]", permute_610: "f32[48, 197, 197]", permute_611: "f32[48, 64, 197]", alias_36: "f32[8, 6, 197, 197]", permute_612: "f32[48, 64, 197]", permute_613: "f32[48, 197, 64]", permute_618: "f32[384, 384]", permute_623: "f32[768, 384]", permute_625: "f32[384, 384]", permute_629: "f32[24, 96]", permute_633: "f32[96, 24]", div_58: "f32[1568, 16, 1]", permute_637: "f32[24, 24]", permute_642: "f32[6272, 16, 16]", permute_643: "f32[6272, 6, 16]", alias_37: "f32[1568, 4, 16, 16]", permute_644: "f32[6272, 6, 16]", permute_645: "f32[6272, 16, 6]", permute_650: "f32[24, 24]", permute_655: "f32[48, 24]", div_59: "f32[1568, 16, 1]", permute_657: "f32[384, 1536]", permute_661: "f32[1536, 384]", div_60: "f32[8, 197, 1]", permute_665: "f32[384, 384]", permute_670: "f32[48, 197, 197]", permute_671: "f32[48, 64, 197]", alias_38: "f32[8, 6, 197, 197]", permute_672: "f32[48, 64, 197]", permute_673: "f32[48, 197, 64]", permute_678: "f32[384, 384]", permute_683: "f32[768, 384]", permute_685: "f32[384, 384]", permute_689: "f32[24, 96]", permute_693: "f32[96, 24]", div_63: "f32[1568, 16, 1]", permute_697: "f32[24, 24]", permute_702: "f32[6272, 16, 16]", permute_703: "f32[6272, 6, 16]", alias_39: "f32[1568, 4, 16, 16]", permute_704: "f32[6272, 6, 16]", permute_705: "f32[6272, 16, 6]", permute_710: "f32[24, 24]", permute_715: "f32[48, 24]", div_64: "f32[1568, 16, 1]", permute_717: "f32[384, 1536]", permute_721: "f32[1536, 384]", div_65: "f32[8, 197, 1]", permute_725: "f32[384, 384]", permute_730: "f32[48, 197, 197]", permute_731: "f32[48, 64, 197]", alias_40: "f32[8, 6, 197, 197]", permute_732: "f32[48, 64, 197]", permute_733: "f32[48, 197, 64]", permute_738: "f32[384, 384]", permute_743: "f32[768, 384]", permute_745: "f32[384, 384]", permute_749: "f32[24, 96]", permute_753: "f32[96, 24]", div_68: "f32[1568, 16, 1]", permute_757: "f32[24, 24]", permute_762: "f32[6272, 16, 16]", permute_763: "f32[6272, 6, 16]", alias_41: "f32[1568, 4, 16, 16]", permute_764: "f32[6272, 6, 16]", permute_765: "f32[6272, 16, 6]", permute_770: "f32[24, 24]", permute_775: "f32[48, 24]", div_69: "f32[1568, 16, 1]", permute_777: "f32[384, 1536]", permute_781: "f32[1536, 384]", div_70: "f32[8, 197, 1]", permute_785: "f32[384, 384]", permute_790: "f32[48, 197, 197]", permute_791: "f32[48, 64, 197]", alias_42: "f32[8, 6, 197, 197]", permute_792: "f32[48, 64, 197]", permute_793: "f32[48, 197, 64]", permute_798: "f32[384, 384]", permute_803: "f32[768, 384]", permute_805: "f32[384, 384]", permute_809: "f32[24, 96]", permute_813: "f32[96, 24]", div_73: "f32[1568, 16, 1]", permute_817: "f32[24, 24]", permute_822: "f32[6272, 16, 16]", permute_823: "f32[6272, 6, 16]", alias_43: "f32[1568, 4, 16, 16]", permute_824: "f32[6272, 6, 16]", permute_825: "f32[6272, 16, 6]", permute_830: "f32[24, 24]", permute_835: "f32[48, 24]", div_74: "f32[1568, 16, 1]", permute_837: "f32[384, 1536]", permute_841: "f32[1536, 384]", div_75: "f32[8, 197, 1]", permute_845: "f32[384, 384]", permute_850: "f32[48, 197, 197]", permute_851: "f32[48, 64, 197]", alias_44: "f32[8, 6, 197, 197]", permute_852: "f32[48, 64, 197]", permute_853: "f32[48, 197, 64]", permute_858: "f32[384, 384]", permute_863: "f32[768, 384]", permute_865: "f32[384, 384]", permute_869: "f32[24, 96]", permute_873: "f32[96, 24]", div_78: "f32[1568, 16, 1]", permute_877: "f32[24, 24]", permute_882: "f32[6272, 16, 16]", permute_883: "f32[6272, 6, 16]", alias_45: "f32[1568, 4, 16, 16]", permute_884: "f32[6272, 6, 16]", permute_885: "f32[6272, 16, 6]", permute_890: "f32[24, 24]", permute_895: "f32[48, 24]", div_79: "f32[1568, 16, 1]", permute_897: "f32[384, 1536]", permute_901: "f32[1536, 384]", div_80: "f32[8, 197, 1]", permute_905: "f32[384, 384]", permute_910: "f32[48, 197, 197]", permute_911: "f32[48, 64, 197]", alias_46: "f32[8, 6, 197, 197]", permute_912: "f32[48, 64, 197]", permute_913: "f32[48, 197, 64]", permute_918: "f32[384, 384]", permute_923: "f32[768, 384]", permute_925: "f32[384, 384]", permute_929: "f32[24, 96]", permute_933: "f32[96, 24]", div_83: "f32[1568, 16, 1]", permute_937: "f32[24, 24]", permute_942: "f32[6272, 16, 16]", permute_943: "f32[6272, 6, 16]", alias_47: "f32[1568, 4, 16, 16]", permute_944: "f32[6272, 6, 16]", permute_945: "f32[6272, 16, 6]", permute_950: "f32[24, 24]", permute_955: "f32[48, 24]", permute_957: "f32[384, 384]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    view_3: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(clone_2, [8, 196, 384])
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_3, getitem_1);  view_3 = getitem_1 = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    view_5: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm, [8, 196, 384]);  addmm = None
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_5, getitem_3);  view_5 = getitem_3 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    sub_2: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_2, getitem_5);  clone_2 = getitem_5 = None
    mul_4: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_2, [1568, 16, 96]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_10: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476)
    erf: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_13: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_6: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_13);  cat_1 = getitem_13 = None
    mul_14: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_44: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_6, [8, 197, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_20: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476)
    erf_1: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_23: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_63: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_9, [1568, 16, 96]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_30: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_13: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_27);  cat_2 = getitem_27 = None
    mul_32: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_13, [8, 197, 1536]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_40: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_104: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_16, [1568, 16, 96]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476)
    erf_4: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
    add_47: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_20: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_3, getitem_41);  cat_3 = getitem_41 = None
    mul_50: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_20, [8, 197, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_56: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476)
    erf_5: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_57: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_23, [1568, 16, 96]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_64: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476)
    erf_6: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_64: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_27: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_4, getitem_55);  cat_4 = getitem_55 = None
    mul_68: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_20);  sub_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_167: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_27, [8, 197, 1536]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
    erf_7: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_74: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_30, [1568, 16, 96]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, 0.7071067811865476)
    erf_8: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_81: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_34: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_5, getitem_69);  cat_5 = getitem_69 = None
    mul_86: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_208: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_92: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, 0.7071067811865476)
    erf_9: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_91: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_227: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_37, [1568, 16, 96]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_100: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476)
    erf_10: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_98: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_41: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_6, getitem_83);  cat_6 = getitem_83 = None
    mul_104: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_30);  sub_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_41, [8, 197, 1536]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_110: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476)
    erf_11: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_108: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_268: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_44, [1568, 16, 96]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_118: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476)
    erf_12: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_115: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_48: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_7, getitem_97);  cat_7 = getitem_97 = None
    mul_122: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_35);  sub_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_290: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_48, [8, 197, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_128: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476)
    erf_13: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_125: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_309: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_51, [1568, 16, 96]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, 0.7071067811865476)
    erf_14: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_132: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_55: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_8, getitem_111);  cat_8 = getitem_111 = None
    mul_140: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_40);  sub_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_331: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_55, [8, 197, 1536]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_146: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, 0.7071067811865476)
    erf_15: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_142: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_350: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_58, [1568, 16, 96]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, 0.7071067811865476)
    erf_16: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_149: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_62: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_9, getitem_125);  cat_9 = getitem_125 = None
    mul_158: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_45);  sub_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_372: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_62, [8, 197, 1536]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_164: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476)
    erf_17: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_159: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_65, [1568, 16, 96]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_172: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476)
    erf_18: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_166: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_69: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_10, getitem_139);  cat_10 = getitem_139 = None
    mul_176: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_50);  sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_413: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_69, [8, 197, 1536]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_182: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, 0.7071067811865476)
    erf_19: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_176: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_72, [1568, 16, 96]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_190: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476)
    erf_20: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
    add_183: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_76: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_11, getitem_153);  cat_11 = getitem_153 = None
    mul_194: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_55);  sub_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_454: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_76, [8, 197, 1536]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_200: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, 0.7071067811865476)
    erf_21: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_200);  mul_200 = None
    add_193: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_473: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(addmm_79, [1568, 16, 96]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, 0.7071067811865476)
    erf_22: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_200: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    sub_83: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_12, getitem_167);  cat_12 = getitem_167 = None
    mul_212: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_60);  sub_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_495: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(addmm_83, [8, 197, 1536]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_218: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, 0.7071067811865476)
    erf_23: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
    add_210: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:330, code: return x if pre_logits else self.head(x)
    mm_48: "f32[8, 384]" = torch.ops.aten.mm.default(tangents_1, permute_233);  permute_233 = None
    permute_234: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_49: "f32[1000, 384]" = torch.ops.aten.mm.default(permute_234, clone_184);  permute_234 = clone_184 = None
    permute_235: "f32[384, 1000]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_25: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_498: "f32[1000]" = torch.ops.aten.reshape.default(sum_25, [1000]);  sum_25 = None
    permute_236: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:328, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_default: "f32[8, 197, 384]" = torch.ops.aten.full.default([8, 197, 384], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 197, 384]" = torch.ops.aten.select_scatter.default(full_default, mm_48, 1, 0);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:323, code: patch_embed = self.norm(patch_embed)
    mul_223: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(select_scatter, primals_348);  primals_348 = None
    mul_224: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_223, 384)
    sum_26: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_223, mul_220);  mul_223 = None
    sum_27: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_220, sum_27);  sum_27 = None
    sub_88: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_224, sum_26);  mul_224 = sum_26 = None
    sub_89: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_88, mul_226);  sub_88 = mul_226 = None
    mul_227: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_24, sub_89);  div_24 = sub_89 = None
    mul_228: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(select_scatter, mul_220);  mul_220 = None
    sum_28: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_29: "f32[384]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_499: "f32[1576, 384]" = torch.ops.aten.reshape.default(mul_227, [1576, 384])
    mm_50: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_499, permute_237);  permute_237 = None
    permute_238: "f32[384, 1576]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_51: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_238, view_496);  permute_238 = view_496 = None
    permute_239: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_30: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
    view_500: "f32[384]" = torch.ops.aten.reshape.default(sum_30, [384]);  sum_30 = None
    permute_240: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_501: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_50, [8, 197, 1536]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_210, 0.5);  add_210 = None
    mul_231: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, view_495)
    mul_232: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_231, -0.5);  mul_231 = None
    exp_24: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_232);  mul_232 = None
    mul_233: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_234: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, mul_233);  view_495 = mul_233 = None
    add_215: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_230, mul_234);  mul_230 = mul_234 = None
    mul_235: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_501, add_215);  view_501 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_502: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_235, [1576, 1536]);  mul_235 = None
    mm_52: "f32[1576, 384]" = torch.ops.aten.mm.default(view_502, permute_241);  permute_241 = None
    permute_242: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_53: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_242, view_494);  permute_242 = view_494 = None
    permute_243: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_31: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
    view_503: "f32[1536]" = torch.ops.aten.reshape.default(sum_31, [1536]);  sum_31 = None
    permute_244: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_504: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_52, [8, 197, 384]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_237: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_504, primals_342);  primals_342 = None
    mul_238: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_237, 384)
    sum_32: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_237, mul_215);  mul_237 = None
    sum_33: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_215, sum_33);  sum_33 = None
    sub_91: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_238, sum_32);  mul_238 = sum_32 = None
    sub_92: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_91, mul_240);  sub_91 = mul_240 = None
    mul_241: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_25, sub_92);  div_25 = sub_92 = None
    mul_242: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_504, mul_215);  mul_215 = None
    sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_504, [0, 1]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_216: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_227, mul_241);  mul_227 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_505: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_216, [1576, 384])
    mm_54: "f32[1576, 384]" = torch.ops.aten.mm.default(view_505, permute_245);  permute_245 = None
    permute_246: "f32[384, 1576]" = torch.ops.aten.permute.default(view_505, [1, 0])
    mm_55: "f32[384, 384]" = torch.ops.aten.mm.default(permute_246, view_492);  permute_246 = view_492 = None
    permute_247: "f32[384, 384]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_36: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True);  view_505 = None
    view_506: "f32[384]" = torch.ops.aten.reshape.default(sum_36, [384]);  sum_36 = None
    permute_248: "f32[384, 384]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_507: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_54, [8, 197, 384]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_508: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_507, [8, 197, 6, 64]);  view_507 = None
    permute_249: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    clone_185: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_509: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_185, [48, 197, 64]);  clone_185 = None
    bmm_48: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_250, view_509);  permute_250 = None
    bmm_49: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_509, permute_251);  view_509 = permute_251 = None
    view_510: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_48, [8, 6, 197, 64]);  bmm_48 = None
    view_511: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_49, [8, 6, 197, 197]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_243: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_511, alias_24);  view_511 = None
    sum_37: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [-1], True)
    mul_244: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_24, sum_37);  alias_24 = sum_37 = None
    sub_93: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_245: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_93, 0.125);  sub_93 = None
    view_512: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_245, [48, 197, 197]);  mul_245 = None
    bmm_50: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_252, view_512);  permute_252 = None
    bmm_51: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_512, permute_253);  view_512 = permute_253 = None
    view_513: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_50, [8, 6, 64, 197]);  bmm_50 = None
    view_514: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 6, 197, 64]);  bmm_51 = None
    permute_254: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_513, [0, 1, 3, 2]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_255: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    clone_186: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_515: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_186, [8, 197, 384]);  clone_186 = None
    view_516: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_515, [1576, 384]);  view_515 = None
    permute_256: "f32[384, 1576]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_56: "f32[384, 384]" = torch.ops.aten.mm.default(permute_256, view_479);  permute_256 = None
    permute_257: "f32[384, 384]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    mm_57: "f32[1576, 384]" = torch.ops.aten.mm.default(view_516, permute_258);  view_516 = permute_258 = None
    view_517: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_57, [8, 197, 384]);  mm_57 = None
    permute_259: "f32[384, 384]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_13: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_514, permute_254]);  view_514 = permute_254 = None
    view_518: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_13, [2, 8, 6, 197, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_260: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_518, [1, 3, 0, 2, 4]);  view_518 = None
    clone_187: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_519: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_187, [8, 197, 768]);  clone_187 = None
    view_520: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_519, [1576, 768]);  view_519 = None
    permute_261: "f32[768, 1576]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_58: "f32[768, 384]" = torch.ops.aten.mm.default(permute_261, view_479);  permute_261 = view_479 = None
    permute_262: "f32[384, 768]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    mm_59: "f32[1576, 384]" = torch.ops.aten.mm.default(view_520, permute_263);  view_520 = permute_263 = None
    view_521: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_59, [8, 197, 384]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_217: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_517, view_521);  view_517 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_264: "f32[768, 384]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_247: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_217, primals_336);  primals_336 = None
    mul_248: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_247, 384)
    sum_38: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_247, mul_212);  mul_247 = None
    sum_39: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_212, sum_39);  sum_39 = None
    sub_95: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_248, sum_38);  mul_248 = sum_38 = None
    sub_96: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_95, mul_250);  sub_95 = mul_250 = None
    div_26: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 384);  rsqrt_60 = None
    mul_251: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_26, sub_96);  div_26 = sub_96 = None
    mul_252: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_217, mul_212);  mul_212 = None
    sum_40: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_217, [0, 1]);  add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_218: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_216, mul_251);  add_216 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_52: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_218, 1, 0, 1)
    slice_53: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_218, 1, 1, 197);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_188: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_53, memory_format = torch.contiguous_format)
    view_522: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_188, [1568, 384]);  clone_188 = None
    mm_60: "f32[1568, 384]" = torch.ops.aten.mm.default(view_522, permute_265);  permute_265 = None
    permute_266: "f32[384, 1568]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_61: "f32[384, 384]" = torch.ops.aten.mm.default(permute_266, view_477);  permute_266 = view_477 = None
    permute_267: "f32[384, 384]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_42: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[384]" = torch.ops.aten.reshape.default(sum_42, [384]);  sum_42 = None
    permute_268: "f32[384, 384]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_524: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_60, [8, 196, 384]);  mm_60 = None
    view_525: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_524, [1568, 16, 24]);  view_524 = None
    mul_254: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_525, primals_332);  primals_332 = None
    mul_255: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_254, 24)
    sum_43: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_254, mul_210);  mul_254 = None
    sum_44: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_210, sum_44);  sum_44 = None
    sub_98: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_255, sum_43);  mul_255 = sum_43 = None
    sub_99: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_98, mul_257);  sub_98 = mul_257 = None
    mul_258: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_27, sub_99);  div_27 = sub_99 = None
    mul_259: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_525, mul_210);  mul_210 = None
    sum_45: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_46: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_525, [0, 1]);  view_525 = None
    slice_scatter_1: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_53, 1, 1, 9223372036854775807);  slice_53 = None
    slice_scatter_3: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_52, 1, 0, 1);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_219: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_1, slice_scatter_3);  slice_scatter_1 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_526: "f32[25088, 24]" = torch.ops.aten.reshape.default(mul_258, [25088, 24])
    mm_62: "f32[25088, 96]" = torch.ops.aten.mm.default(view_526, permute_269);  permute_269 = None
    permute_270: "f32[24, 25088]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_63: "f32[24, 96]" = torch.ops.aten.mm.default(permute_270, view_474);  permute_270 = view_474 = None
    permute_271: "f32[96, 24]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_47: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[24]" = torch.ops.aten.reshape.default(sum_47, [24]);  sum_47 = None
    permute_272: "f32[24, 96]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_528: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_62, [1568, 16, 96]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_261: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_200, 0.5);  add_200 = None
    mul_262: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, view_473)
    mul_263: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_262, -0.5);  mul_262 = None
    exp_25: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_263);  mul_263 = None
    mul_264: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_265: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, mul_264);  view_473 = mul_264 = None
    add_221: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
    mul_266: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_528, add_221);  view_528 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_529: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_266, [25088, 96]);  mul_266 = None
    mm_64: "f32[25088, 24]" = torch.ops.aten.mm.default(view_529, permute_273);  permute_273 = None
    permute_274: "f32[96, 25088]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_65: "f32[96, 24]" = torch.ops.aten.mm.default(permute_274, view_472);  permute_274 = view_472 = None
    permute_275: "f32[24, 96]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_48: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[96]" = torch.ops.aten.reshape.default(sum_48, [96]);  sum_48 = None
    permute_276: "f32[96, 24]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_531: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_64, [1568, 16, 24]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_268: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_531, primals_326);  primals_326 = None
    mul_269: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_268, 24)
    sum_49: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True)
    mul_270: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_268, mul_205);  mul_268 = None
    sum_50: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True);  mul_270 = None
    mul_271: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_205, sum_50);  sum_50 = None
    sub_101: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_269, sum_49);  mul_269 = sum_49 = None
    sub_102: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_101, mul_271);  sub_101 = mul_271 = None
    mul_272: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_28, sub_102);  div_28 = sub_102 = None
    mul_273: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_531, mul_205);  mul_205 = None
    sum_51: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 1]);  mul_273 = None
    sum_52: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_531, [0, 1]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_222: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_258, mul_272);  mul_258 = mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_532: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_222, [25088, 24])
    mm_66: "f32[25088, 24]" = torch.ops.aten.mm.default(view_532, permute_277);  permute_277 = None
    permute_278: "f32[24, 25088]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_67: "f32[24, 24]" = torch.ops.aten.mm.default(permute_278, view_470);  permute_278 = view_470 = None
    permute_279: "f32[24, 24]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_53: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[24]" = torch.ops.aten.reshape.default(sum_53, [24]);  sum_53 = None
    permute_280: "f32[24, 24]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_534: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_66, [1568, 16, 24]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_535: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_534, [1568, 16, 4, 6]);  view_534 = None
    permute_281: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    clone_191: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
    view_536: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_191, [6272, 16, 6]);  clone_191 = None
    bmm_52: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_282, view_536);  permute_282 = None
    bmm_53: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_536, permute_283);  view_536 = permute_283 = None
    view_537: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_52, [1568, 4, 16, 6]);  bmm_52 = None
    view_538: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_53, [1568, 4, 16, 16]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_274: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_538, alias_25);  view_538 = None
    sum_54: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [-1], True)
    mul_275: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_25, sum_54);  alias_25 = sum_54 = None
    sub_103: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_276: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_103, 0.408248290463863);  sub_103 = None
    view_539: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_276, [6272, 16, 16]);  mul_276 = None
    bmm_54: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_284, view_539);  permute_284 = None
    bmm_55: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_539, permute_285);  view_539 = permute_285 = None
    view_540: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_54, [1568, 4, 6, 16]);  bmm_54 = None
    view_541: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_55, [1568, 4, 16, 6]);  bmm_55 = None
    permute_286: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_287: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    clone_192: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_542: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_192, [1568, 16, 24]);  clone_192 = None
    view_543: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_542, [25088, 24]);  view_542 = None
    permute_288: "f32[24, 25088]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_68: "f32[24, 24]" = torch.ops.aten.mm.default(permute_288, view_457);  permute_288 = None
    permute_289: "f32[24, 24]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    mm_69: "f32[25088, 24]" = torch.ops.aten.mm.default(view_543, permute_290);  view_543 = permute_290 = None
    view_544: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_69, [1568, 16, 24]);  mm_69 = None
    permute_291: "f32[24, 24]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_14: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_541, permute_286]);  view_541 = permute_286 = None
    view_545: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_14, [2, 1568, 4, 16, 6]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_292: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_545, [1, 3, 0, 2, 4]);  view_545 = None
    clone_193: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_546: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_193, [1568, 16, 48]);  clone_193 = None
    view_547: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_546, [25088, 48]);  view_546 = None
    permute_293: "f32[48, 25088]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_70: "f32[48, 24]" = torch.ops.aten.mm.default(permute_293, view_457);  permute_293 = view_457 = None
    permute_294: "f32[24, 48]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    mm_71: "f32[25088, 24]" = torch.ops.aten.mm.default(view_547, permute_295);  view_547 = permute_295 = None
    view_548: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_71, [1568, 16, 24]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_223: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_544, view_548);  view_544 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_296: "f32[48, 24]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_278: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_223, primals_320);  primals_320 = None
    mul_279: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_278, 24)
    sum_55: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
    mul_280: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_278, mul_192);  mul_278 = None
    sum_56: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
    mul_281: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_192, sum_56);  sum_56 = None
    sub_105: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_279, sum_55);  mul_279 = sum_55 = None
    sub_106: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_105, mul_281);  sub_105 = mul_281 = None
    mul_282: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_29, sub_106);  sub_106 = None
    mul_283: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_223, mul_192)
    sum_57: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
    sum_58: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_224: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_222, mul_282);  add_222 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_549: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_219, [1576, 384])
    mm_72: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_549, permute_297);  permute_297 = None
    permute_298: "f32[384, 1576]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_73: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_298, view_455);  permute_298 = view_455 = None
    permute_299: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_59: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[384]" = torch.ops.aten.reshape.default(sum_59, [384]);  sum_59 = None
    permute_300: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    view_551: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_72, [8, 197, 1536]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_285: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
    mul_286: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, view_454)
    mul_287: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
    exp_26: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
    mul_288: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_289: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, mul_288);  view_454 = mul_288 = None
    add_226: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
    mul_290: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_551, add_226);  view_551 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_552: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_290, [1576, 1536]);  mul_290 = None
    mm_74: "f32[1576, 384]" = torch.ops.aten.mm.default(view_552, permute_301);  permute_301 = None
    permute_302: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_75: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_302, view_453);  permute_302 = view_453 = None
    permute_303: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_60: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[1536]" = torch.ops.aten.reshape.default(sum_60, [1536]);  sum_60 = None
    permute_304: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    view_554: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_74, [8, 197, 384]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_292: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_554, primals_314);  primals_314 = None
    mul_293: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_292, 384)
    sum_61: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
    mul_294: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_292, mul_197);  mul_292 = None
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
    mul_295: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_197, sum_62);  sum_62 = None
    sub_108: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_293, sum_61);  mul_293 = sum_61 = None
    sub_109: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_108, mul_295);  sub_108 = mul_295 = None
    mul_296: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_30, sub_109);  div_30 = sub_109 = None
    mul_297: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_554, mul_197);  mul_197 = None
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
    sum_64: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_554, [0, 1]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_227: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_219, mul_296);  add_219 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_555: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_227, [1576, 384])
    mm_76: "f32[1576, 384]" = torch.ops.aten.mm.default(view_555, permute_305);  permute_305 = None
    permute_306: "f32[384, 1576]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_77: "f32[384, 384]" = torch.ops.aten.mm.default(permute_306, view_451);  permute_306 = view_451 = None
    permute_307: "f32[384, 384]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_65: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[384]" = torch.ops.aten.reshape.default(sum_65, [384]);  sum_65 = None
    permute_308: "f32[384, 384]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    view_557: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_76, [8, 197, 384]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_558: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_557, [8, 197, 6, 64]);  view_557 = None
    permute_309: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    clone_195: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_559: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_195, [48, 197, 64]);  clone_195 = None
    bmm_56: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_310, view_559);  permute_310 = None
    bmm_57: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_559, permute_311);  view_559 = permute_311 = None
    view_560: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_56, [8, 6, 197, 64]);  bmm_56 = None
    view_561: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_57, [8, 6, 197, 197]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_298: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_561, alias_26);  view_561 = None
    sum_66: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [-1], True)
    mul_299: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_26, sum_66);  alias_26 = sum_66 = None
    sub_110: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_300: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_110, 0.125);  sub_110 = None
    view_562: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_300, [48, 197, 197]);  mul_300 = None
    bmm_58: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_312, view_562);  permute_312 = None
    bmm_59: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_562, permute_313);  view_562 = permute_313 = None
    view_563: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_58, [8, 6, 64, 197]);  bmm_58 = None
    view_564: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 6, 197, 64]);  bmm_59 = None
    permute_314: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_563, [0, 1, 3, 2]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_315: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    clone_196: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
    view_565: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_196, [8, 197, 384]);  clone_196 = None
    view_566: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_565, [1576, 384]);  view_565 = None
    permute_316: "f32[384, 1576]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_78: "f32[384, 384]" = torch.ops.aten.mm.default(permute_316, view_438);  permute_316 = None
    permute_317: "f32[384, 384]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    mm_79: "f32[1576, 384]" = torch.ops.aten.mm.default(view_566, permute_318);  view_566 = permute_318 = None
    view_567: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_79, [8, 197, 384]);  mm_79 = None
    permute_319: "f32[384, 384]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_15: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_564, permute_314]);  view_564 = permute_314 = None
    view_568: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_15, [2, 8, 6, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_320: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_568, [1, 3, 0, 2, 4]);  view_568 = None
    clone_197: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_569: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_197, [8, 197, 768]);  clone_197 = None
    view_570: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_569, [1576, 768]);  view_569 = None
    permute_321: "f32[768, 1576]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_80: "f32[768, 384]" = torch.ops.aten.mm.default(permute_321, view_438);  permute_321 = view_438 = None
    permute_322: "f32[384, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    mm_81: "f32[1576, 384]" = torch.ops.aten.mm.default(view_570, permute_323);  view_570 = permute_323 = None
    view_571: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_81, [8, 197, 384]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_228: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_567, view_571);  view_567 = view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_324: "f32[768, 384]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_302: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_228, primals_308);  primals_308 = None
    mul_303: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_302, 384)
    sum_67: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
    mul_304: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_302, mul_194);  mul_302 = None
    sum_68: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
    mul_305: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_194, sum_68);  sum_68 = None
    sub_112: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_303, sum_67);  mul_303 = sum_67 = None
    sub_113: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_112, mul_305);  sub_112 = mul_305 = None
    div_31: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 384);  rsqrt_55 = None
    mul_306: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_31, sub_113);  div_31 = sub_113 = None
    mul_307: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_228, mul_194);  mul_194 = None
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
    sum_70: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_228, [0, 1]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_229: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_227, mul_306);  add_227 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_54: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_229, 1, 0, 1)
    slice_55: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_229, 1, 1, 197);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_55, memory_format = torch.contiguous_format)
    view_572: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_198, [1568, 384]);  clone_198 = None
    mm_82: "f32[1568, 384]" = torch.ops.aten.mm.default(view_572, permute_325);  permute_325 = None
    permute_326: "f32[384, 1568]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_83: "f32[384, 384]" = torch.ops.aten.mm.default(permute_326, view_436);  permute_326 = view_436 = None
    permute_327: "f32[384, 384]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_71: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[384]" = torch.ops.aten.reshape.default(sum_71, [384]);  sum_71 = None
    permute_328: "f32[384, 384]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    view_574: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_82, [8, 196, 384]);  mm_82 = None
    view_575: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_574, [1568, 16, 24]);  view_574 = None
    mul_309: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_575, primals_304);  primals_304 = None
    mul_310: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_309, 24)
    sum_72: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_309, mul_192);  mul_309 = None
    sum_73: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_192, sum_73);  sum_73 = None
    sub_115: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_310, sum_72);  mul_310 = sum_72 = None
    sub_116: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_115, mul_312);  sub_115 = mul_312 = None
    mul_313: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_29, sub_116);  div_29 = sub_116 = None
    mul_314: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_575, mul_192);  mul_192 = None
    sum_74: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_75: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_575, [0, 1]);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_230: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_224, mul_313);  add_224 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_5: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_55, 1, 1, 9223372036854775807);  slice_55 = None
    slice_scatter_7: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_54, 1, 0, 1);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_231: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_5, slice_scatter_7);  slice_scatter_5 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_576: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_230, [25088, 24])
    mm_84: "f32[25088, 96]" = torch.ops.aten.mm.default(view_576, permute_329);  permute_329 = None
    permute_330: "f32[24, 25088]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_85: "f32[24, 96]" = torch.ops.aten.mm.default(permute_330, view_433);  permute_330 = view_433 = None
    permute_331: "f32[96, 24]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_76: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[24]" = torch.ops.aten.reshape.default(sum_76, [24]);  sum_76 = None
    permute_332: "f32[24, 96]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_578: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_84, [1568, 16, 96]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_316: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_183, 0.5);  add_183 = None
    mul_317: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, view_432)
    mul_318: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
    exp_27: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_318);  mul_318 = None
    mul_319: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_320: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, mul_319);  view_432 = mul_319 = None
    add_233: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
    mul_321: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_578, add_233);  view_578 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_579: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_321, [25088, 96]);  mul_321 = None
    mm_86: "f32[25088, 24]" = torch.ops.aten.mm.default(view_579, permute_333);  permute_333 = None
    permute_334: "f32[96, 25088]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_87: "f32[96, 24]" = torch.ops.aten.mm.default(permute_334, view_431);  permute_334 = view_431 = None
    permute_335: "f32[24, 96]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_77: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[96]" = torch.ops.aten.reshape.default(sum_77, [96]);  sum_77 = None
    permute_336: "f32[96, 24]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_581: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_86, [1568, 16, 24]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_323: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_581, primals_298);  primals_298 = None
    mul_324: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_323, 24)
    sum_78: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
    mul_325: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_323, mul_187);  mul_323 = None
    sum_79: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
    mul_326: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_187, sum_79);  sum_79 = None
    sub_118: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_324, sum_78);  mul_324 = sum_78 = None
    sub_119: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_118, mul_326);  sub_118 = mul_326 = None
    mul_327: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_33, sub_119);  div_33 = sub_119 = None
    mul_328: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_581, mul_187);  mul_187 = None
    sum_80: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
    sum_81: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_581, [0, 1]);  view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_234: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_230, mul_327);  add_230 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_582: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_234, [25088, 24])
    mm_88: "f32[25088, 24]" = torch.ops.aten.mm.default(view_582, permute_337);  permute_337 = None
    permute_338: "f32[24, 25088]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_89: "f32[24, 24]" = torch.ops.aten.mm.default(permute_338, view_429);  permute_338 = view_429 = None
    permute_339: "f32[24, 24]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_82: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[24]" = torch.ops.aten.reshape.default(sum_82, [24]);  sum_82 = None
    permute_340: "f32[24, 24]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_584: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_88, [1568, 16, 24]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_585: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_584, [1568, 16, 4, 6]);  view_584 = None
    permute_341: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_585, [0, 2, 1, 3]);  view_585 = None
    clone_201: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None
    view_586: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_201, [6272, 16, 6]);  clone_201 = None
    bmm_60: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_342, view_586);  permute_342 = None
    bmm_61: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_586, permute_343);  view_586 = permute_343 = None
    view_587: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_60, [1568, 4, 16, 6]);  bmm_60 = None
    view_588: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_61, [1568, 4, 16, 16]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_329: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_588, alias_27);  view_588 = None
    sum_83: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [-1], True)
    mul_330: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_27, sum_83);  alias_27 = sum_83 = None
    sub_120: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_331: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, 0.408248290463863);  sub_120 = None
    view_589: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_331, [6272, 16, 16]);  mul_331 = None
    bmm_62: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_344, view_589);  permute_344 = None
    bmm_63: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_589, permute_345);  view_589 = permute_345 = None
    view_590: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_62, [1568, 4, 6, 16]);  bmm_62 = None
    view_591: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_63, [1568, 4, 16, 6]);  bmm_63 = None
    permute_346: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_590, [0, 1, 3, 2]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_347: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_587, [0, 2, 1, 3]);  view_587 = None
    clone_202: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    view_592: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_202, [1568, 16, 24]);  clone_202 = None
    view_593: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_592, [25088, 24]);  view_592 = None
    permute_348: "f32[24, 25088]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_90: "f32[24, 24]" = torch.ops.aten.mm.default(permute_348, view_416);  permute_348 = None
    permute_349: "f32[24, 24]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    mm_91: "f32[25088, 24]" = torch.ops.aten.mm.default(view_593, permute_350);  view_593 = permute_350 = None
    view_594: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_91, [1568, 16, 24]);  mm_91 = None
    permute_351: "f32[24, 24]" = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_16: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_591, permute_346]);  view_591 = permute_346 = None
    view_595: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_16, [2, 1568, 4, 16, 6]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_352: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_595, [1, 3, 0, 2, 4]);  view_595 = None
    clone_203: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
    view_596: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_203, [1568, 16, 48]);  clone_203 = None
    view_597: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_596, [25088, 48]);  view_596 = None
    permute_353: "f32[48, 25088]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_92: "f32[48, 24]" = torch.ops.aten.mm.default(permute_353, view_416);  permute_353 = view_416 = None
    permute_354: "f32[24, 48]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    mm_93: "f32[25088, 24]" = torch.ops.aten.mm.default(view_597, permute_355);  view_597 = permute_355 = None
    view_598: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_93, [1568, 16, 24]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_235: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_594, view_598);  view_594 = view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_356: "f32[48, 24]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_333: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_235, primals_292);  primals_292 = None
    mul_334: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_333, 24)
    sum_84: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
    mul_335: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_333, mul_174);  mul_333 = None
    sum_85: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
    mul_336: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_174, sum_85);  sum_85 = None
    sub_122: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_334, sum_84);  mul_334 = sum_84 = None
    sub_123: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_122, mul_336);  sub_122 = mul_336 = None
    mul_337: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_34, sub_123);  sub_123 = None
    mul_338: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_235, mul_174)
    sum_86: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
    sum_87: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_235, [0, 1]);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_236: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_234, mul_337);  add_234 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_599: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_231, [1576, 384])
    mm_94: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_599, permute_357);  permute_357 = None
    permute_358: "f32[384, 1576]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_95: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_358, view_414);  permute_358 = view_414 = None
    permute_359: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_88: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True);  view_599 = None
    view_600: "f32[384]" = torch.ops.aten.reshape.default(sum_88, [384]);  sum_88 = None
    permute_360: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    view_601: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_94, [8, 197, 1536]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_340: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_341: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, view_413)
    mul_342: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_28: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_344: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, mul_343);  view_413 = mul_343 = None
    add_238: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_601, add_238);  view_601 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_602: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_345, [1576, 1536]);  mul_345 = None
    mm_96: "f32[1576, 384]" = torch.ops.aten.mm.default(view_602, permute_361);  permute_361 = None
    permute_362: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_97: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_362, view_412);  permute_362 = view_412 = None
    permute_363: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_89: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
    view_603: "f32[1536]" = torch.ops.aten.reshape.default(sum_89, [1536]);  sum_89 = None
    permute_364: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_604: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_96, [8, 197, 384]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_347: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_604, primals_286);  primals_286 = None
    mul_348: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_347, 384)
    sum_90: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_347, mul_179);  mul_347 = None
    sum_91: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_179, sum_91);  sum_91 = None
    sub_125: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_348, sum_90);  mul_348 = sum_90 = None
    sub_126: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_125, mul_350);  sub_125 = mul_350 = None
    mul_351: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_35, sub_126);  div_35 = sub_126 = None
    mul_352: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_604, mul_179);  mul_179 = None
    sum_92: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_93: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_604, [0, 1]);  view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_239: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_231, mul_351);  add_231 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_605: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_239, [1576, 384])
    mm_98: "f32[1576, 384]" = torch.ops.aten.mm.default(view_605, permute_365);  permute_365 = None
    permute_366: "f32[384, 1576]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_99: "f32[384, 384]" = torch.ops.aten.mm.default(permute_366, view_410);  permute_366 = view_410 = None
    permute_367: "f32[384, 384]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_94: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
    view_606: "f32[384]" = torch.ops.aten.reshape.default(sum_94, [384]);  sum_94 = None
    permute_368: "f32[384, 384]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_607: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_98, [8, 197, 384]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_608: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_607, [8, 197, 6, 64]);  view_607 = None
    permute_369: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
    clone_205: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
    view_609: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_205, [48, 197, 64]);  clone_205 = None
    bmm_64: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_370, view_609);  permute_370 = None
    bmm_65: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_609, permute_371);  view_609 = permute_371 = None
    view_610: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_64, [8, 6, 197, 64]);  bmm_64 = None
    view_611: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_65, [8, 6, 197, 197]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_353: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_611, alias_28);  view_611 = None
    sum_95: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
    mul_354: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_28, sum_95);  alias_28 = sum_95 = None
    sub_127: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_355: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_127, 0.125);  sub_127 = None
    view_612: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_355, [48, 197, 197]);  mul_355 = None
    bmm_66: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_372, view_612);  permute_372 = None
    bmm_67: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_612, permute_373);  view_612 = permute_373 = None
    view_613: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_66, [8, 6, 64, 197]);  bmm_66 = None
    view_614: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 6, 197, 64]);  bmm_67 = None
    permute_374: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_613, [0, 1, 3, 2]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_375: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
    clone_206: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    view_615: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_206, [8, 197, 384]);  clone_206 = None
    view_616: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_615, [1576, 384]);  view_615 = None
    permute_376: "f32[384, 1576]" = torch.ops.aten.permute.default(view_616, [1, 0])
    mm_100: "f32[384, 384]" = torch.ops.aten.mm.default(permute_376, view_397);  permute_376 = None
    permute_377: "f32[384, 384]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    mm_101: "f32[1576, 384]" = torch.ops.aten.mm.default(view_616, permute_378);  view_616 = permute_378 = None
    view_617: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_101, [8, 197, 384]);  mm_101 = None
    permute_379: "f32[384, 384]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_17: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_614, permute_374]);  view_614 = permute_374 = None
    view_618: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_17, [2, 8, 6, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_380: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_618, [1, 3, 0, 2, 4]);  view_618 = None
    clone_207: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
    view_619: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_207, [8, 197, 768]);  clone_207 = None
    view_620: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_619, [1576, 768]);  view_619 = None
    permute_381: "f32[768, 1576]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_102: "f32[768, 384]" = torch.ops.aten.mm.default(permute_381, view_397);  permute_381 = view_397 = None
    permute_382: "f32[384, 768]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    mm_103: "f32[1576, 384]" = torch.ops.aten.mm.default(view_620, permute_383);  view_620 = permute_383 = None
    view_621: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_103, [8, 197, 384]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_240: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_617, view_621);  view_617 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_384: "f32[768, 384]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_357: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_240, primals_280);  primals_280 = None
    mul_358: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_357, 384)
    sum_96: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_357, mul_176);  mul_357 = None
    sum_97: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_176, sum_97);  sum_97 = None
    sub_129: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_358, sum_96);  mul_358 = sum_96 = None
    sub_130: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_129, mul_360);  sub_129 = mul_360 = None
    div_36: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 384);  rsqrt_50 = None
    mul_361: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_36, sub_130);  div_36 = sub_130 = None
    mul_362: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_240, mul_176);  mul_176 = None
    sum_98: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_99: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_240, [0, 1]);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_241: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_239, mul_361);  add_239 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_56: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_241, 1, 0, 1)
    slice_57: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_241, 1, 1, 197);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_208: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_57, memory_format = torch.contiguous_format)
    view_622: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_208, [1568, 384]);  clone_208 = None
    mm_104: "f32[1568, 384]" = torch.ops.aten.mm.default(view_622, permute_385);  permute_385 = None
    permute_386: "f32[384, 1568]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_105: "f32[384, 384]" = torch.ops.aten.mm.default(permute_386, view_395);  permute_386 = view_395 = None
    permute_387: "f32[384, 384]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_100: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
    view_623: "f32[384]" = torch.ops.aten.reshape.default(sum_100, [384]);  sum_100 = None
    permute_388: "f32[384, 384]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_624: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_104, [8, 196, 384]);  mm_104 = None
    view_625: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_624, [1568, 16, 24]);  view_624 = None
    mul_364: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_625, primals_276);  primals_276 = None
    mul_365: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_364, 24)
    sum_101: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
    mul_366: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_364, mul_174);  mul_364 = None
    sum_102: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
    mul_367: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_174, sum_102);  sum_102 = None
    sub_132: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_365, sum_101);  mul_365 = sum_101 = None
    sub_133: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_132, mul_367);  sub_132 = mul_367 = None
    mul_368: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_34, sub_133);  div_34 = sub_133 = None
    mul_369: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_625, mul_174);  mul_174 = None
    sum_103: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
    sum_104: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_625, [0, 1]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_242: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_236, mul_368);  add_236 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_9: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_57, 1, 1, 9223372036854775807);  slice_57 = None
    slice_scatter_11: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_56, 1, 0, 1);  slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_243: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_9, slice_scatter_11);  slice_scatter_9 = slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_626: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_242, [25088, 24])
    mm_106: "f32[25088, 96]" = torch.ops.aten.mm.default(view_626, permute_389);  permute_389 = None
    permute_390: "f32[24, 25088]" = torch.ops.aten.permute.default(view_626, [1, 0])
    mm_107: "f32[24, 96]" = torch.ops.aten.mm.default(permute_390, view_392);  permute_390 = view_392 = None
    permute_391: "f32[96, 24]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_105: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_626, [0], True);  view_626 = None
    view_627: "f32[24]" = torch.ops.aten.reshape.default(sum_105, [24]);  sum_105 = None
    permute_392: "f32[24, 96]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    view_628: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_106, [1568, 16, 96]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_371: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_372: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, view_391)
    mul_373: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_372, -0.5);  mul_372 = None
    exp_29: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_373);  mul_373 = None
    mul_374: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_375: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, mul_374);  view_391 = mul_374 = None
    add_245: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_371, mul_375);  mul_371 = mul_375 = None
    mul_376: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_628, add_245);  view_628 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_629: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_376, [25088, 96]);  mul_376 = None
    mm_108: "f32[25088, 24]" = torch.ops.aten.mm.default(view_629, permute_393);  permute_393 = None
    permute_394: "f32[96, 25088]" = torch.ops.aten.permute.default(view_629, [1, 0])
    mm_109: "f32[96, 24]" = torch.ops.aten.mm.default(permute_394, view_390);  permute_394 = view_390 = None
    permute_395: "f32[24, 96]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_106: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_629, [0], True);  view_629 = None
    view_630: "f32[96]" = torch.ops.aten.reshape.default(sum_106, [96]);  sum_106 = None
    permute_396: "f32[96, 24]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_631: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_108, [1568, 16, 24]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_378: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_631, primals_270);  primals_270 = None
    mul_379: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_378, 24)
    sum_107: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True)
    mul_380: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_378, mul_169);  mul_378 = None
    sum_108: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
    mul_381: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_169, sum_108);  sum_108 = None
    sub_135: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_379, sum_107);  mul_379 = sum_107 = None
    sub_136: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_135, mul_381);  sub_135 = mul_381 = None
    mul_382: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_38, sub_136);  div_38 = sub_136 = None
    mul_383: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_631, mul_169);  mul_169 = None
    sum_109: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1]);  mul_383 = None
    sum_110: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_631, [0, 1]);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_246: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_242, mul_382);  add_242 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_632: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_246, [25088, 24])
    mm_110: "f32[25088, 24]" = torch.ops.aten.mm.default(view_632, permute_397);  permute_397 = None
    permute_398: "f32[24, 25088]" = torch.ops.aten.permute.default(view_632, [1, 0])
    mm_111: "f32[24, 24]" = torch.ops.aten.mm.default(permute_398, view_388);  permute_398 = view_388 = None
    permute_399: "f32[24, 24]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_111: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_632, [0], True);  view_632 = None
    view_633: "f32[24]" = torch.ops.aten.reshape.default(sum_111, [24]);  sum_111 = None
    permute_400: "f32[24, 24]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    view_634: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_110, [1568, 16, 24]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_635: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_634, [1568, 16, 4, 6]);  view_634 = None
    permute_401: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
    clone_211: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
    view_636: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_211, [6272, 16, 6]);  clone_211 = None
    bmm_68: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_402, view_636);  permute_402 = None
    bmm_69: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_636, permute_403);  view_636 = permute_403 = None
    view_637: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_68, [1568, 4, 16, 6]);  bmm_68 = None
    view_638: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_69, [1568, 4, 16, 16]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_384: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_638, alias_29);  view_638 = None
    sum_112: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [-1], True)
    mul_385: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_29, sum_112);  alias_29 = sum_112 = None
    sub_137: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_386: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_137, 0.408248290463863);  sub_137 = None
    view_639: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_386, [6272, 16, 16]);  mul_386 = None
    bmm_70: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_404, view_639);  permute_404 = None
    bmm_71: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_639, permute_405);  view_639 = permute_405 = None
    view_640: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_70, [1568, 4, 6, 16]);  bmm_70 = None
    view_641: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_71, [1568, 4, 16, 6]);  bmm_71 = None
    permute_406: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_640, [0, 1, 3, 2]);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_407: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
    clone_212: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    view_642: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_212, [1568, 16, 24]);  clone_212 = None
    view_643: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_642, [25088, 24]);  view_642 = None
    permute_408: "f32[24, 25088]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_112: "f32[24, 24]" = torch.ops.aten.mm.default(permute_408, view_375);  permute_408 = None
    permute_409: "f32[24, 24]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    mm_113: "f32[25088, 24]" = torch.ops.aten.mm.default(view_643, permute_410);  view_643 = permute_410 = None
    view_644: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_113, [1568, 16, 24]);  mm_113 = None
    permute_411: "f32[24, 24]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_18: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_641, permute_406]);  view_641 = permute_406 = None
    view_645: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_18, [2, 1568, 4, 16, 6]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_412: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_645, [1, 3, 0, 2, 4]);  view_645 = None
    clone_213: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_646: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_213, [1568, 16, 48]);  clone_213 = None
    view_647: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_646, [25088, 48]);  view_646 = None
    permute_413: "f32[48, 25088]" = torch.ops.aten.permute.default(view_647, [1, 0])
    mm_114: "f32[48, 24]" = torch.ops.aten.mm.default(permute_413, view_375);  permute_413 = view_375 = None
    permute_414: "f32[24, 48]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    mm_115: "f32[25088, 24]" = torch.ops.aten.mm.default(view_647, permute_415);  view_647 = permute_415 = None
    view_648: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_115, [1568, 16, 24]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_247: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_644, view_648);  view_644 = view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_416: "f32[48, 24]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_388: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_247, primals_264);  primals_264 = None
    mul_389: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_388, 24)
    sum_113: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_388, mul_156);  mul_388 = None
    sum_114: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_156, sum_114);  sum_114 = None
    sub_139: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_389, sum_113);  mul_389 = sum_113 = None
    sub_140: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_139, mul_391);  sub_139 = mul_391 = None
    mul_392: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_39, sub_140);  sub_140 = None
    mul_393: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_247, mul_156)
    sum_115: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_116: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_247, [0, 1]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_248: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_246, mul_392);  add_246 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_649: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_243, [1576, 384])
    mm_116: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_649, permute_417);  permute_417 = None
    permute_418: "f32[384, 1576]" = torch.ops.aten.permute.default(view_649, [1, 0])
    mm_117: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_418, view_373);  permute_418 = view_373 = None
    permute_419: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_117: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True);  view_649 = None
    view_650: "f32[384]" = torch.ops.aten.reshape.default(sum_117, [384]);  sum_117 = None
    permute_420: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
    view_651: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_116, [8, 197, 1536]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_395: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_159, 0.5);  add_159 = None
    mul_396: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, view_372)
    mul_397: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
    exp_30: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
    mul_398: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_399: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, mul_398);  view_372 = mul_398 = None
    add_250: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
    mul_400: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_651, add_250);  view_651 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_652: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_400, [1576, 1536]);  mul_400 = None
    mm_118: "f32[1576, 384]" = torch.ops.aten.mm.default(view_652, permute_421);  permute_421 = None
    permute_422: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_652, [1, 0])
    mm_119: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_422, view_371);  permute_422 = view_371 = None
    permute_423: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_118: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_652, [0], True);  view_652 = None
    view_653: "f32[1536]" = torch.ops.aten.reshape.default(sum_118, [1536]);  sum_118 = None
    permute_424: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_654: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_118, [8, 197, 384]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_402: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_654, primals_258);  primals_258 = None
    mul_403: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_402, 384)
    sum_119: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
    mul_404: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_402, mul_161);  mul_402 = None
    sum_120: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
    mul_405: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_161, sum_120);  sum_120 = None
    sub_142: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_403, sum_119);  mul_403 = sum_119 = None
    sub_143: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_142, mul_405);  sub_142 = mul_405 = None
    mul_406: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_40, sub_143);  div_40 = sub_143 = None
    mul_407: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_654, mul_161);  mul_161 = None
    sum_121: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
    sum_122: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_654, [0, 1]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_251: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_243, mul_406);  add_243 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_655: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_251, [1576, 384])
    mm_120: "f32[1576, 384]" = torch.ops.aten.mm.default(view_655, permute_425);  permute_425 = None
    permute_426: "f32[384, 1576]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_121: "f32[384, 384]" = torch.ops.aten.mm.default(permute_426, view_369);  permute_426 = view_369 = None
    permute_427: "f32[384, 384]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_123: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_655, [0], True);  view_655 = None
    view_656: "f32[384]" = torch.ops.aten.reshape.default(sum_123, [384]);  sum_123 = None
    permute_428: "f32[384, 384]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_657: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_120, [8, 197, 384]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_658: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_657, [8, 197, 6, 64]);  view_657 = None
    permute_429: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
    clone_215: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
    view_659: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_215, [48, 197, 64]);  clone_215 = None
    bmm_72: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_430, view_659);  permute_430 = None
    bmm_73: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_659, permute_431);  view_659 = permute_431 = None
    view_660: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_72, [8, 6, 197, 64]);  bmm_72 = None
    view_661: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_73, [8, 6, 197, 197]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_408: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_661, alias_30);  view_661 = None
    sum_124: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [-1], True)
    mul_409: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_30, sum_124);  alias_30 = sum_124 = None
    sub_144: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_410: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_144, 0.125);  sub_144 = None
    view_662: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_410, [48, 197, 197]);  mul_410 = None
    bmm_74: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_432, view_662);  permute_432 = None
    bmm_75: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_662, permute_433);  view_662 = permute_433 = None
    view_663: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_74, [8, 6, 64, 197]);  bmm_74 = None
    view_664: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_75, [8, 6, 197, 64]);  bmm_75 = None
    permute_434: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_663, [0, 1, 3, 2]);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_435: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_660, [0, 2, 1, 3]);  view_660 = None
    clone_216: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    view_665: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_216, [8, 197, 384]);  clone_216 = None
    view_666: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_665, [1576, 384]);  view_665 = None
    permute_436: "f32[384, 1576]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_122: "f32[384, 384]" = torch.ops.aten.mm.default(permute_436, view_356);  permute_436 = None
    permute_437: "f32[384, 384]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    mm_123: "f32[1576, 384]" = torch.ops.aten.mm.default(view_666, permute_438);  view_666 = permute_438 = None
    view_667: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_123, [8, 197, 384]);  mm_123 = None
    permute_439: "f32[384, 384]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_19: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_664, permute_434]);  view_664 = permute_434 = None
    view_668: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_19, [2, 8, 6, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_440: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_668, [1, 3, 0, 2, 4]);  view_668 = None
    clone_217: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
    view_669: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_217, [8, 197, 768]);  clone_217 = None
    view_670: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_669, [1576, 768]);  view_669 = None
    permute_441: "f32[768, 1576]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_124: "f32[768, 384]" = torch.ops.aten.mm.default(permute_441, view_356);  permute_441 = view_356 = None
    permute_442: "f32[384, 768]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    mm_125: "f32[1576, 384]" = torch.ops.aten.mm.default(view_670, permute_443);  view_670 = permute_443 = None
    view_671: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_125, [8, 197, 384]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_252: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_667, view_671);  view_667 = view_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_444: "f32[768, 384]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_412: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_252, primals_252);  primals_252 = None
    mul_413: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_412, 384)
    sum_125: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_412, mul_158);  mul_412 = None
    sum_126: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_158, sum_126);  sum_126 = None
    sub_146: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_413, sum_125);  mul_413 = sum_125 = None
    sub_147: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_146, mul_415);  sub_146 = mul_415 = None
    div_41: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 384);  rsqrt_45 = None
    mul_416: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_41, sub_147);  div_41 = sub_147 = None
    mul_417: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_252, mul_158);  mul_158 = None
    sum_127: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_128: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_253: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_251, mul_416);  add_251 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_58: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_253, 1, 0, 1)
    slice_59: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_253, 1, 1, 197);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_218: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_59, memory_format = torch.contiguous_format)
    view_672: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_218, [1568, 384]);  clone_218 = None
    mm_126: "f32[1568, 384]" = torch.ops.aten.mm.default(view_672, permute_445);  permute_445 = None
    permute_446: "f32[384, 1568]" = torch.ops.aten.permute.default(view_672, [1, 0])
    mm_127: "f32[384, 384]" = torch.ops.aten.mm.default(permute_446, view_354);  permute_446 = view_354 = None
    permute_447: "f32[384, 384]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_129: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_672, [0], True);  view_672 = None
    view_673: "f32[384]" = torch.ops.aten.reshape.default(sum_129, [384]);  sum_129 = None
    permute_448: "f32[384, 384]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    view_674: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_126, [8, 196, 384]);  mm_126 = None
    view_675: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_674, [1568, 16, 24]);  view_674 = None
    mul_419: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_675, primals_248);  primals_248 = None
    mul_420: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_419, 24)
    sum_130: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
    mul_421: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_419, mul_156);  mul_419 = None
    sum_131: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
    mul_422: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_156, sum_131);  sum_131 = None
    sub_149: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_420, sum_130);  mul_420 = sum_130 = None
    sub_150: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_149, mul_422);  sub_149 = mul_422 = None
    mul_423: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_39, sub_150);  div_39 = sub_150 = None
    mul_424: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_675, mul_156);  mul_156 = None
    sum_132: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
    sum_133: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_675, [0, 1]);  view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_254: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_248, mul_423);  add_248 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_13: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_59, 1, 1, 9223372036854775807);  slice_59 = None
    slice_scatter_15: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_58, 1, 0, 1);  slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_255: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_13, slice_scatter_15);  slice_scatter_13 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_676: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_254, [25088, 24])
    mm_128: "f32[25088, 96]" = torch.ops.aten.mm.default(view_676, permute_449);  permute_449 = None
    permute_450: "f32[24, 25088]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_129: "f32[24, 96]" = torch.ops.aten.mm.default(permute_450, view_351);  permute_450 = view_351 = None
    permute_451: "f32[96, 24]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_134: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
    view_677: "f32[24]" = torch.ops.aten.reshape.default(sum_134, [24]);  sum_134 = None
    permute_452: "f32[24, 96]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
    view_678: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_128, [1568, 16, 96]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_426: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_149, 0.5);  add_149 = None
    mul_427: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, view_350)
    mul_428: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_427, -0.5);  mul_427 = None
    exp_31: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_428);  mul_428 = None
    mul_429: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_430: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, mul_429);  view_350 = mul_429 = None
    add_257: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_426, mul_430);  mul_426 = mul_430 = None
    mul_431: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_678, add_257);  view_678 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_679: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_431, [25088, 96]);  mul_431 = None
    mm_130: "f32[25088, 24]" = torch.ops.aten.mm.default(view_679, permute_453);  permute_453 = None
    permute_454: "f32[96, 25088]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_131: "f32[96, 24]" = torch.ops.aten.mm.default(permute_454, view_349);  permute_454 = view_349 = None
    permute_455: "f32[24, 96]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_135: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True);  view_679 = None
    view_680: "f32[96]" = torch.ops.aten.reshape.default(sum_135, [96]);  sum_135 = None
    permute_456: "f32[96, 24]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    view_681: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_130, [1568, 16, 24]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_433: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_681, primals_242);  primals_242 = None
    mul_434: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_433, 24)
    sum_136: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True)
    mul_435: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_433, mul_151);  mul_433 = None
    sum_137: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True);  mul_435 = None
    mul_436: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_151, sum_137);  sum_137 = None
    sub_152: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_434, sum_136);  mul_434 = sum_136 = None
    sub_153: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_152, mul_436);  sub_152 = mul_436 = None
    mul_437: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_43, sub_153);  div_43 = sub_153 = None
    mul_438: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_681, mul_151);  mul_151 = None
    sum_138: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1]);  mul_438 = None
    sum_139: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_681, [0, 1]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_258: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_254, mul_437);  add_254 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_682: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_258, [25088, 24])
    mm_132: "f32[25088, 24]" = torch.ops.aten.mm.default(view_682, permute_457);  permute_457 = None
    permute_458: "f32[24, 25088]" = torch.ops.aten.permute.default(view_682, [1, 0])
    mm_133: "f32[24, 24]" = torch.ops.aten.mm.default(permute_458, view_347);  permute_458 = view_347 = None
    permute_459: "f32[24, 24]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_140: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_682, [0], True);  view_682 = None
    view_683: "f32[24]" = torch.ops.aten.reshape.default(sum_140, [24]);  sum_140 = None
    permute_460: "f32[24, 24]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    view_684: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_132, [1568, 16, 24]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_685: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_684, [1568, 16, 4, 6]);  view_684 = None
    permute_461: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_685, [0, 2, 1, 3]);  view_685 = None
    clone_221: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
    view_686: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_221, [6272, 16, 6]);  clone_221 = None
    bmm_76: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_462, view_686);  permute_462 = None
    bmm_77: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_686, permute_463);  view_686 = permute_463 = None
    view_687: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_76, [1568, 4, 16, 6]);  bmm_76 = None
    view_688: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_77, [1568, 4, 16, 16]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_439: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_688, alias_31);  view_688 = None
    sum_141: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [-1], True)
    mul_440: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_31, sum_141);  alias_31 = sum_141 = None
    sub_154: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_441: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_154, 0.408248290463863);  sub_154 = None
    view_689: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_441, [6272, 16, 16]);  mul_441 = None
    bmm_78: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_464, view_689);  permute_464 = None
    bmm_79: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_689, permute_465);  view_689 = permute_465 = None
    view_690: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_78, [1568, 4, 6, 16]);  bmm_78 = None
    view_691: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_79, [1568, 4, 16, 6]);  bmm_79 = None
    permute_466: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_690, [0, 1, 3, 2]);  view_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_467: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
    clone_222: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    view_692: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_222, [1568, 16, 24]);  clone_222 = None
    view_693: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_692, [25088, 24]);  view_692 = None
    permute_468: "f32[24, 25088]" = torch.ops.aten.permute.default(view_693, [1, 0])
    mm_134: "f32[24, 24]" = torch.ops.aten.mm.default(permute_468, view_334);  permute_468 = None
    permute_469: "f32[24, 24]" = torch.ops.aten.permute.default(mm_134, [1, 0]);  mm_134 = None
    mm_135: "f32[25088, 24]" = torch.ops.aten.mm.default(view_693, permute_470);  view_693 = permute_470 = None
    view_694: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_135, [1568, 16, 24]);  mm_135 = None
    permute_471: "f32[24, 24]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_20: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_691, permute_466]);  view_691 = permute_466 = None
    view_695: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_20, [2, 1568, 4, 16, 6]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_472: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_695, [1, 3, 0, 2, 4]);  view_695 = None
    clone_223: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_472, memory_format = torch.contiguous_format);  permute_472 = None
    view_696: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_223, [1568, 16, 48]);  clone_223 = None
    view_697: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_696, [25088, 48]);  view_696 = None
    permute_473: "f32[48, 25088]" = torch.ops.aten.permute.default(view_697, [1, 0])
    mm_136: "f32[48, 24]" = torch.ops.aten.mm.default(permute_473, view_334);  permute_473 = view_334 = None
    permute_474: "f32[24, 48]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[25088, 24]" = torch.ops.aten.mm.default(view_697, permute_475);  view_697 = permute_475 = None
    view_698: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_137, [1568, 16, 24]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_259: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_694, view_698);  view_694 = view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_476: "f32[48, 24]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_443: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_259, primals_236);  primals_236 = None
    mul_444: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_443, 24)
    sum_142: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
    mul_445: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_443, mul_138);  mul_443 = None
    sum_143: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
    mul_446: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_138, sum_143);  sum_143 = None
    sub_156: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_444, sum_142);  mul_444 = sum_142 = None
    sub_157: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_156, mul_446);  sub_156 = mul_446 = None
    mul_447: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_44, sub_157);  sub_157 = None
    mul_448: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_259, mul_138)
    sum_144: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
    sum_145: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_259, [0, 1]);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_260: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_258, mul_447);  add_258 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_699: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_255, [1576, 384])
    mm_138: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_699, permute_477);  permute_477 = None
    permute_478: "f32[384, 1576]" = torch.ops.aten.permute.default(view_699, [1, 0])
    mm_139: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_478, view_332);  permute_478 = view_332 = None
    permute_479: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_146: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_699, [0], True);  view_699 = None
    view_700: "f32[384]" = torch.ops.aten.reshape.default(sum_146, [384]);  sum_146 = None
    permute_480: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_701: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_138, [8, 197, 1536]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_450: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_451: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, view_331)
    mul_452: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_451, -0.5);  mul_451 = None
    exp_32: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_452);  mul_452 = None
    mul_453: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_454: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, mul_453);  view_331 = mul_453 = None
    add_262: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_450, mul_454);  mul_450 = mul_454 = None
    mul_455: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_701, add_262);  view_701 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_702: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_455, [1576, 1536]);  mul_455 = None
    mm_140: "f32[1576, 384]" = torch.ops.aten.mm.default(view_702, permute_481);  permute_481 = None
    permute_482: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_141: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_482, view_330);  permute_482 = view_330 = None
    permute_483: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_147: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
    view_703: "f32[1536]" = torch.ops.aten.reshape.default(sum_147, [1536]);  sum_147 = None
    permute_484: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_704: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_140, [8, 197, 384]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_457: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_704, primals_230);  primals_230 = None
    mul_458: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_457, 384)
    sum_148: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True)
    mul_459: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_457, mul_143);  mul_457 = None
    sum_149: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True);  mul_459 = None
    mul_460: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_143, sum_149);  sum_149 = None
    sub_159: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_458, sum_148);  mul_458 = sum_148 = None
    sub_160: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_159, mul_460);  sub_159 = mul_460 = None
    mul_461: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_45, sub_160);  div_45 = sub_160 = None
    mul_462: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_704, mul_143);  mul_143 = None
    sum_150: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 1]);  mul_462 = None
    sum_151: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_704, [0, 1]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_263: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_255, mul_461);  add_255 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_705: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_263, [1576, 384])
    mm_142: "f32[1576, 384]" = torch.ops.aten.mm.default(view_705, permute_485);  permute_485 = None
    permute_486: "f32[384, 1576]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_143: "f32[384, 384]" = torch.ops.aten.mm.default(permute_486, view_328);  permute_486 = view_328 = None
    permute_487: "f32[384, 384]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_152: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[384]" = torch.ops.aten.reshape.default(sum_152, [384]);  sum_152 = None
    permute_488: "f32[384, 384]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_707: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_142, [8, 197, 384]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_708: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_707, [8, 197, 6, 64]);  view_707 = None
    permute_489: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None
    clone_225: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_709: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_225, [48, 197, 64]);  clone_225 = None
    bmm_80: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_490, view_709);  permute_490 = None
    bmm_81: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_709, permute_491);  view_709 = permute_491 = None
    view_710: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_80, [8, 6, 197, 64]);  bmm_80 = None
    view_711: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_81, [8, 6, 197, 197]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_463: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_711, alias_32);  view_711 = None
    sum_153: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [-1], True)
    mul_464: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_32, sum_153);  alias_32 = sum_153 = None
    sub_161: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_465: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_161, 0.125);  sub_161 = None
    view_712: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_465, [48, 197, 197]);  mul_465 = None
    bmm_82: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_492, view_712);  permute_492 = None
    bmm_83: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_712, permute_493);  view_712 = permute_493 = None
    view_713: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_82, [8, 6, 64, 197]);  bmm_82 = None
    view_714: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_83, [8, 6, 197, 64]);  bmm_83 = None
    permute_494: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_713, [0, 1, 3, 2]);  view_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_495: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
    clone_226: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
    view_715: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_226, [8, 197, 384]);  clone_226 = None
    view_716: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_715, [1576, 384]);  view_715 = None
    permute_496: "f32[384, 1576]" = torch.ops.aten.permute.default(view_716, [1, 0])
    mm_144: "f32[384, 384]" = torch.ops.aten.mm.default(permute_496, view_315);  permute_496 = None
    permute_497: "f32[384, 384]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[1576, 384]" = torch.ops.aten.mm.default(view_716, permute_498);  view_716 = permute_498 = None
    view_717: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_145, [8, 197, 384]);  mm_145 = None
    permute_499: "f32[384, 384]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_21: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_714, permute_494]);  view_714 = permute_494 = None
    view_718: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_21, [2, 8, 6, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_500: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_718, [1, 3, 0, 2, 4]);  view_718 = None
    clone_227: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
    view_719: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_227, [8, 197, 768]);  clone_227 = None
    view_720: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_719, [1576, 768]);  view_719 = None
    permute_501: "f32[768, 1576]" = torch.ops.aten.permute.default(view_720, [1, 0])
    mm_146: "f32[768, 384]" = torch.ops.aten.mm.default(permute_501, view_315);  permute_501 = view_315 = None
    permute_502: "f32[384, 768]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    mm_147: "f32[1576, 384]" = torch.ops.aten.mm.default(view_720, permute_503);  view_720 = permute_503 = None
    view_721: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_147, [8, 197, 384]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_264: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_717, view_721);  view_717 = view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_504: "f32[768, 384]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_467: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_264, primals_224);  primals_224 = None
    mul_468: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_467, 384)
    sum_154: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_467, mul_140);  mul_467 = None
    sum_155: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_140, sum_155);  sum_155 = None
    sub_163: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_468, sum_154);  mul_468 = sum_154 = None
    sub_164: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_163, mul_470);  sub_163 = mul_470 = None
    div_46: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 384);  rsqrt_40 = None
    mul_471: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_46, sub_164);  div_46 = sub_164 = None
    mul_472: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_264, mul_140);  mul_140 = None
    sum_156: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_157: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_264, [0, 1]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_265: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_263, mul_471);  add_263 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_60: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_265, 1, 0, 1)
    slice_61: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_265, 1, 1, 197);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_228: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_61, memory_format = torch.contiguous_format)
    view_722: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_228, [1568, 384]);  clone_228 = None
    mm_148: "f32[1568, 384]" = torch.ops.aten.mm.default(view_722, permute_505);  permute_505 = None
    permute_506: "f32[384, 1568]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_149: "f32[384, 384]" = torch.ops.aten.mm.default(permute_506, view_313);  permute_506 = view_313 = None
    permute_507: "f32[384, 384]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_158: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[384]" = torch.ops.aten.reshape.default(sum_158, [384]);  sum_158 = None
    permute_508: "f32[384, 384]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_724: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_148, [8, 196, 384]);  mm_148 = None
    view_725: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_724, [1568, 16, 24]);  view_724 = None
    mul_474: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_725, primals_220);  primals_220 = None
    mul_475: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_474, 24)
    sum_159: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
    mul_476: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_474, mul_138);  mul_474 = None
    sum_160: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
    mul_477: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_138, sum_160);  sum_160 = None
    sub_166: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_475, sum_159);  mul_475 = sum_159 = None
    sub_167: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_166, mul_477);  sub_166 = mul_477 = None
    mul_478: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_44, sub_167);  div_44 = sub_167 = None
    mul_479: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_725, mul_138);  mul_138 = None
    sum_161: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
    sum_162: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_725, [0, 1]);  view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_266: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_260, mul_478);  add_260 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_17: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_61, 1, 1, 9223372036854775807);  slice_61 = None
    slice_scatter_19: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_60, 1, 0, 1);  slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_267: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_17, slice_scatter_19);  slice_scatter_17 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_726: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_266, [25088, 24])
    mm_150: "f32[25088, 96]" = torch.ops.aten.mm.default(view_726, permute_509);  permute_509 = None
    permute_510: "f32[24, 25088]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_151: "f32[24, 96]" = torch.ops.aten.mm.default(permute_510, view_310);  permute_510 = view_310 = None
    permute_511: "f32[96, 24]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_163: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[24]" = torch.ops.aten.reshape.default(sum_163, [24]);  sum_163 = None
    permute_512: "f32[24, 96]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_728: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_150, [1568, 16, 96]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_481: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_482: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, view_309)
    mul_483: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_482, -0.5);  mul_482 = None
    exp_33: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_483);  mul_483 = None
    mul_484: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_485: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, mul_484);  view_309 = mul_484 = None
    add_269: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_481, mul_485);  mul_481 = mul_485 = None
    mul_486: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_728, add_269);  view_728 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_729: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_486, [25088, 96]);  mul_486 = None
    mm_152: "f32[25088, 24]" = torch.ops.aten.mm.default(view_729, permute_513);  permute_513 = None
    permute_514: "f32[96, 25088]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_153: "f32[96, 24]" = torch.ops.aten.mm.default(permute_514, view_308);  permute_514 = view_308 = None
    permute_515: "f32[24, 96]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_164: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[96]" = torch.ops.aten.reshape.default(sum_164, [96]);  sum_164 = None
    permute_516: "f32[96, 24]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_731: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_152, [1568, 16, 24]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_488: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_731, primals_214);  primals_214 = None
    mul_489: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_488, 24)
    sum_165: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True)
    mul_490: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_488, mul_133);  mul_488 = None
    sum_166: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
    mul_491: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_133, sum_166);  sum_166 = None
    sub_169: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_489, sum_165);  mul_489 = sum_165 = None
    sub_170: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_169, mul_491);  sub_169 = mul_491 = None
    mul_492: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_48, sub_170);  div_48 = sub_170 = None
    mul_493: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_731, mul_133);  mul_133 = None
    sum_167: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1]);  mul_493 = None
    sum_168: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_731, [0, 1]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_270: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_266, mul_492);  add_266 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_732: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_270, [25088, 24])
    mm_154: "f32[25088, 24]" = torch.ops.aten.mm.default(view_732, permute_517);  permute_517 = None
    permute_518: "f32[24, 25088]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_155: "f32[24, 24]" = torch.ops.aten.mm.default(permute_518, view_306);  permute_518 = view_306 = None
    permute_519: "f32[24, 24]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_169: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_732, [0], True);  view_732 = None
    view_733: "f32[24]" = torch.ops.aten.reshape.default(sum_169, [24]);  sum_169 = None
    permute_520: "f32[24, 24]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_734: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_154, [1568, 16, 24]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_735: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_734, [1568, 16, 4, 6]);  view_734 = None
    permute_521: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
    clone_231: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_521, memory_format = torch.contiguous_format);  permute_521 = None
    view_736: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_231, [6272, 16, 6]);  clone_231 = None
    bmm_84: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_522, view_736);  permute_522 = None
    bmm_85: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_736, permute_523);  view_736 = permute_523 = None
    view_737: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_84, [1568, 4, 16, 6]);  bmm_84 = None
    view_738: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_85, [1568, 4, 16, 16]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_494: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_738, alias_33);  view_738 = None
    sum_170: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [-1], True)
    mul_495: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_33, sum_170);  alias_33 = sum_170 = None
    sub_171: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_496: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_171, 0.408248290463863);  sub_171 = None
    view_739: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_496, [6272, 16, 16]);  mul_496 = None
    bmm_86: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_524, view_739);  permute_524 = None
    bmm_87: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_739, permute_525);  view_739 = permute_525 = None
    view_740: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_86, [1568, 4, 6, 16]);  bmm_86 = None
    view_741: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_87, [1568, 4, 16, 6]);  bmm_87 = None
    permute_526: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_740, [0, 1, 3, 2]);  view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_527: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_737, [0, 2, 1, 3]);  view_737 = None
    clone_232: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
    view_742: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_232, [1568, 16, 24]);  clone_232 = None
    view_743: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_742, [25088, 24]);  view_742 = None
    permute_528: "f32[24, 25088]" = torch.ops.aten.permute.default(view_743, [1, 0])
    mm_156: "f32[24, 24]" = torch.ops.aten.mm.default(permute_528, view_293);  permute_528 = None
    permute_529: "f32[24, 24]" = torch.ops.aten.permute.default(mm_156, [1, 0]);  mm_156 = None
    mm_157: "f32[25088, 24]" = torch.ops.aten.mm.default(view_743, permute_530);  view_743 = permute_530 = None
    view_744: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_157, [1568, 16, 24]);  mm_157 = None
    permute_531: "f32[24, 24]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_22: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_741, permute_526]);  view_741 = permute_526 = None
    view_745: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_22, [2, 1568, 4, 16, 6]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_532: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_745, [1, 3, 0, 2, 4]);  view_745 = None
    clone_233: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_532, memory_format = torch.contiguous_format);  permute_532 = None
    view_746: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_233, [1568, 16, 48]);  clone_233 = None
    view_747: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_746, [25088, 48]);  view_746 = None
    permute_533: "f32[48, 25088]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_158: "f32[48, 24]" = torch.ops.aten.mm.default(permute_533, view_293);  permute_533 = view_293 = None
    permute_534: "f32[24, 48]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    mm_159: "f32[25088, 24]" = torch.ops.aten.mm.default(view_747, permute_535);  view_747 = permute_535 = None
    view_748: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_159, [1568, 16, 24]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_271: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_744, view_748);  view_744 = view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_536: "f32[48, 24]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_498: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_271, primals_208);  primals_208 = None
    mul_499: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_498, 24)
    sum_171: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True)
    mul_500: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_498, mul_120);  mul_498 = None
    sum_172: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
    mul_501: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_120, sum_172);  sum_172 = None
    sub_173: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_499, sum_171);  mul_499 = sum_171 = None
    sub_174: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_173, mul_501);  sub_173 = mul_501 = None
    mul_502: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_49, sub_174);  sub_174 = None
    mul_503: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_271, mul_120)
    sum_173: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
    sum_174: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_271, [0, 1]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_272: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_270, mul_502);  add_270 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_749: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_267, [1576, 384])
    mm_160: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_749, permute_537);  permute_537 = None
    permute_538: "f32[384, 1576]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_161: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_538, view_291);  permute_538 = view_291 = None
    permute_539: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_175: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[384]" = torch.ops.aten.reshape.default(sum_175, [384]);  sum_175 = None
    permute_540: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    view_751: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_160, [8, 197, 1536]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_505: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_506: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, view_290)
    mul_507: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_506, -0.5);  mul_506 = None
    exp_34: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_507);  mul_507 = None
    mul_508: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_509: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, mul_508);  view_290 = mul_508 = None
    add_274: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_505, mul_509);  mul_505 = mul_509 = None
    mul_510: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_751, add_274);  view_751 = add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_752: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_510, [1576, 1536]);  mul_510 = None
    mm_162: "f32[1576, 384]" = torch.ops.aten.mm.default(view_752, permute_541);  permute_541 = None
    permute_542: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_752, [1, 0])
    mm_163: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_542, view_289);  permute_542 = view_289 = None
    permute_543: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_176: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_752, [0], True);  view_752 = None
    view_753: "f32[1536]" = torch.ops.aten.reshape.default(sum_176, [1536]);  sum_176 = None
    permute_544: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    view_754: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_162, [8, 197, 384]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_512: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_754, primals_202);  primals_202 = None
    mul_513: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_512, 384)
    sum_177: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True)
    mul_514: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_512, mul_125);  mul_512 = None
    sum_178: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True);  mul_514 = None
    mul_515: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_125, sum_178);  sum_178 = None
    sub_176: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_513, sum_177);  mul_513 = sum_177 = None
    sub_177: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_176, mul_515);  sub_176 = mul_515 = None
    mul_516: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_50, sub_177);  div_50 = sub_177 = None
    mul_517: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_754, mul_125);  mul_125 = None
    sum_179: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1]);  mul_517 = None
    sum_180: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_754, [0, 1]);  view_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_275: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_267, mul_516);  add_267 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_755: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_275, [1576, 384])
    mm_164: "f32[1576, 384]" = torch.ops.aten.mm.default(view_755, permute_545);  permute_545 = None
    permute_546: "f32[384, 1576]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_165: "f32[384, 384]" = torch.ops.aten.mm.default(permute_546, view_287);  permute_546 = view_287 = None
    permute_547: "f32[384, 384]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_181: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[384]" = torch.ops.aten.reshape.default(sum_181, [384]);  sum_181 = None
    permute_548: "f32[384, 384]" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None
    view_757: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_164, [8, 197, 384]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_758: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_757, [8, 197, 6, 64]);  view_757 = None
    permute_549: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
    clone_235: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_549, memory_format = torch.contiguous_format);  permute_549 = None
    view_759: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_235, [48, 197, 64]);  clone_235 = None
    bmm_88: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_550, view_759);  permute_550 = None
    bmm_89: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_759, permute_551);  view_759 = permute_551 = None
    view_760: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_88, [8, 6, 197, 64]);  bmm_88 = None
    view_761: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_89, [8, 6, 197, 197]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_518: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_761, alias_34);  view_761 = None
    sum_182: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [-1], True)
    mul_519: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_34, sum_182);  alias_34 = sum_182 = None
    sub_178: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_520: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_178, 0.125);  sub_178 = None
    view_762: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_520, [48, 197, 197]);  mul_520 = None
    bmm_90: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_552, view_762);  permute_552 = None
    bmm_91: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_762, permute_553);  view_762 = permute_553 = None
    view_763: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_90, [8, 6, 64, 197]);  bmm_90 = None
    view_764: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_91, [8, 6, 197, 64]);  bmm_91 = None
    permute_554: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_763, [0, 1, 3, 2]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_555: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
    clone_236: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_555, memory_format = torch.contiguous_format);  permute_555 = None
    view_765: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_236, [8, 197, 384]);  clone_236 = None
    view_766: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_765, [1576, 384]);  view_765 = None
    permute_556: "f32[384, 1576]" = torch.ops.aten.permute.default(view_766, [1, 0])
    mm_166: "f32[384, 384]" = torch.ops.aten.mm.default(permute_556, view_274);  permute_556 = None
    permute_557: "f32[384, 384]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[1576, 384]" = torch.ops.aten.mm.default(view_766, permute_558);  view_766 = permute_558 = None
    view_767: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_167, [8, 197, 384]);  mm_167 = None
    permute_559: "f32[384, 384]" = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_23: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_764, permute_554]);  view_764 = permute_554 = None
    view_768: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_23, [2, 8, 6, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_560: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_768, [1, 3, 0, 2, 4]);  view_768 = None
    clone_237: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
    view_769: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_237, [8, 197, 768]);  clone_237 = None
    view_770: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_769, [1576, 768]);  view_769 = None
    permute_561: "f32[768, 1576]" = torch.ops.aten.permute.default(view_770, [1, 0])
    mm_168: "f32[768, 384]" = torch.ops.aten.mm.default(permute_561, view_274);  permute_561 = view_274 = None
    permute_562: "f32[384, 768]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[1576, 384]" = torch.ops.aten.mm.default(view_770, permute_563);  view_770 = permute_563 = None
    view_771: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_169, [8, 197, 384]);  mm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_276: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_767, view_771);  view_767 = view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_564: "f32[768, 384]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_522: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_276, primals_196);  primals_196 = None
    mul_523: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_522, 384)
    sum_183: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
    mul_524: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_522, mul_122);  mul_522 = None
    sum_184: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
    mul_525: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_122, sum_184);  sum_184 = None
    sub_180: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_523, sum_183);  mul_523 = sum_183 = None
    sub_181: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_180, mul_525);  sub_180 = mul_525 = None
    div_51: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 384);  rsqrt_35 = None
    mul_526: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_51, sub_181);  div_51 = sub_181 = None
    mul_527: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_276, mul_122);  mul_122 = None
    sum_185: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
    sum_186: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_276, [0, 1]);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_277: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_275, mul_526);  add_275 = mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_62: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_277, 1, 0, 1)
    slice_63: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_277, 1, 1, 197);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_238: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_63, memory_format = torch.contiguous_format)
    view_772: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_238, [1568, 384]);  clone_238 = None
    mm_170: "f32[1568, 384]" = torch.ops.aten.mm.default(view_772, permute_565);  permute_565 = None
    permute_566: "f32[384, 1568]" = torch.ops.aten.permute.default(view_772, [1, 0])
    mm_171: "f32[384, 384]" = torch.ops.aten.mm.default(permute_566, view_272);  permute_566 = view_272 = None
    permute_567: "f32[384, 384]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_187: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_772, [0], True);  view_772 = None
    view_773: "f32[384]" = torch.ops.aten.reshape.default(sum_187, [384]);  sum_187 = None
    permute_568: "f32[384, 384]" = torch.ops.aten.permute.default(permute_567, [1, 0]);  permute_567 = None
    view_774: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_170, [8, 196, 384]);  mm_170 = None
    view_775: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_774, [1568, 16, 24]);  view_774 = None
    mul_529: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_775, primals_192);  primals_192 = None
    mul_530: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_529, 24)
    sum_188: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True)
    mul_531: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_529, mul_120);  mul_529 = None
    sum_189: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True);  mul_531 = None
    mul_532: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_120, sum_189);  sum_189 = None
    sub_183: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_530, sum_188);  mul_530 = sum_188 = None
    sub_184: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_183, mul_532);  sub_183 = mul_532 = None
    mul_533: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_49, sub_184);  div_49 = sub_184 = None
    mul_534: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_775, mul_120);  mul_120 = None
    sum_190: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 1]);  mul_534 = None
    sum_191: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_775, [0, 1]);  view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_278: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_272, mul_533);  add_272 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_21: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_63, 1, 1, 9223372036854775807);  slice_63 = None
    slice_scatter_23: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_62, 1, 0, 1);  slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_279: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_21, slice_scatter_23);  slice_scatter_21 = slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_776: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_278, [25088, 24])
    mm_172: "f32[25088, 96]" = torch.ops.aten.mm.default(view_776, permute_569);  permute_569 = None
    permute_570: "f32[24, 25088]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_173: "f32[24, 96]" = torch.ops.aten.mm.default(permute_570, view_269);  permute_570 = view_269 = None
    permute_571: "f32[96, 24]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_192: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[24]" = torch.ops.aten.reshape.default(sum_192, [24]);  sum_192 = None
    permute_572: "f32[24, 96]" = torch.ops.aten.permute.default(permute_571, [1, 0]);  permute_571 = None
    view_778: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_172, [1568, 16, 96]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_536: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_537: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, view_268)
    mul_538: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_537, -0.5);  mul_537 = None
    exp_35: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_538);  mul_538 = None
    mul_539: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_540: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, mul_539);  view_268 = mul_539 = None
    add_281: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_536, mul_540);  mul_536 = mul_540 = None
    mul_541: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_778, add_281);  view_778 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_779: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_541, [25088, 96]);  mul_541 = None
    mm_174: "f32[25088, 24]" = torch.ops.aten.mm.default(view_779, permute_573);  permute_573 = None
    permute_574: "f32[96, 25088]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_175: "f32[96, 24]" = torch.ops.aten.mm.default(permute_574, view_267);  permute_574 = view_267 = None
    permute_575: "f32[24, 96]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_193: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[96]" = torch.ops.aten.reshape.default(sum_193, [96]);  sum_193 = None
    permute_576: "f32[96, 24]" = torch.ops.aten.permute.default(permute_575, [1, 0]);  permute_575 = None
    view_781: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_174, [1568, 16, 24]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_543: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_781, primals_186);  primals_186 = None
    mul_544: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_543, 24)
    sum_194: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True)
    mul_545: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_543, mul_115);  mul_543 = None
    sum_195: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True);  mul_545 = None
    mul_546: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_115, sum_195);  sum_195 = None
    sub_186: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_544, sum_194);  mul_544 = sum_194 = None
    sub_187: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_186, mul_546);  sub_186 = mul_546 = None
    mul_547: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_53, sub_187);  div_53 = sub_187 = None
    mul_548: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_781, mul_115);  mul_115 = None
    sum_196: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 1]);  mul_548 = None
    sum_197: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_781, [0, 1]);  view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_282: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_278, mul_547);  add_278 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_782: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_282, [25088, 24])
    mm_176: "f32[25088, 24]" = torch.ops.aten.mm.default(view_782, permute_577);  permute_577 = None
    permute_578: "f32[24, 25088]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_177: "f32[24, 24]" = torch.ops.aten.mm.default(permute_578, view_265);  permute_578 = view_265 = None
    permute_579: "f32[24, 24]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_198: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
    view_783: "f32[24]" = torch.ops.aten.reshape.default(sum_198, [24]);  sum_198 = None
    permute_580: "f32[24, 24]" = torch.ops.aten.permute.default(permute_579, [1, 0]);  permute_579 = None
    view_784: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_176, [1568, 16, 24]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_785: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_784, [1568, 16, 4, 6]);  view_784 = None
    permute_581: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_785, [0, 2, 1, 3]);  view_785 = None
    clone_241: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
    view_786: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_241, [6272, 16, 6]);  clone_241 = None
    bmm_92: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_582, view_786);  permute_582 = None
    bmm_93: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_786, permute_583);  view_786 = permute_583 = None
    view_787: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_92, [1568, 4, 16, 6]);  bmm_92 = None
    view_788: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_93, [1568, 4, 16, 16]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_549: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_788, alias_35);  view_788 = None
    sum_199: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [-1], True)
    mul_550: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_35, sum_199);  alias_35 = sum_199 = None
    sub_188: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_551: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_188, 0.408248290463863);  sub_188 = None
    view_789: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_551, [6272, 16, 16]);  mul_551 = None
    bmm_94: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_584, view_789);  permute_584 = None
    bmm_95: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_789, permute_585);  view_789 = permute_585 = None
    view_790: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_94, [1568, 4, 6, 16]);  bmm_94 = None
    view_791: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_95, [1568, 4, 16, 6]);  bmm_95 = None
    permute_586: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_790, [0, 1, 3, 2]);  view_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_587: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_787, [0, 2, 1, 3]);  view_787 = None
    clone_242: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
    view_792: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_242, [1568, 16, 24]);  clone_242 = None
    view_793: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_792, [25088, 24]);  view_792 = None
    permute_588: "f32[24, 25088]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_178: "f32[24, 24]" = torch.ops.aten.mm.default(permute_588, view_252);  permute_588 = None
    permute_589: "f32[24, 24]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    mm_179: "f32[25088, 24]" = torch.ops.aten.mm.default(view_793, permute_590);  view_793 = permute_590 = None
    view_794: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_179, [1568, 16, 24]);  mm_179 = None
    permute_591: "f32[24, 24]" = torch.ops.aten.permute.default(permute_589, [1, 0]);  permute_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_24: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_791, permute_586]);  view_791 = permute_586 = None
    view_795: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_24, [2, 1568, 4, 16, 6]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_592: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_795, [1, 3, 0, 2, 4]);  view_795 = None
    clone_243: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_592, memory_format = torch.contiguous_format);  permute_592 = None
    view_796: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_243, [1568, 16, 48]);  clone_243 = None
    view_797: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_796, [25088, 48]);  view_796 = None
    permute_593: "f32[48, 25088]" = torch.ops.aten.permute.default(view_797, [1, 0])
    mm_180: "f32[48, 24]" = torch.ops.aten.mm.default(permute_593, view_252);  permute_593 = view_252 = None
    permute_594: "f32[24, 48]" = torch.ops.aten.permute.default(mm_180, [1, 0]);  mm_180 = None
    mm_181: "f32[25088, 24]" = torch.ops.aten.mm.default(view_797, permute_595);  view_797 = permute_595 = None
    view_798: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_181, [1568, 16, 24]);  mm_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_283: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_794, view_798);  view_794 = view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_596: "f32[48, 24]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_553: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_283, primals_180);  primals_180 = None
    mul_554: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_553, 24)
    sum_200: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True)
    mul_555: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_553, mul_102);  mul_553 = None
    sum_201: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True);  mul_555 = None
    mul_556: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_102, sum_201);  sum_201 = None
    sub_190: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_554, sum_200);  mul_554 = sum_200 = None
    sub_191: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_190, mul_556);  sub_190 = mul_556 = None
    mul_557: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_54, sub_191);  sub_191 = None
    mul_558: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_283, mul_102)
    sum_202: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 1]);  mul_558 = None
    sum_203: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_283, [0, 1]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_284: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_282, mul_557);  add_282 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_799: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_279, [1576, 384])
    mm_182: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_799, permute_597);  permute_597 = None
    permute_598: "f32[384, 1576]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_183: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_598, view_250);  permute_598 = view_250 = None
    permute_599: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_204: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[384]" = torch.ops.aten.reshape.default(sum_204, [384]);  sum_204 = None
    permute_600: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
    view_801: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_182, [8, 197, 1536]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_560: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_561: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, view_249)
    mul_562: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_561, -0.5);  mul_561 = None
    exp_36: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_562);  mul_562 = None
    mul_563: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_564: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, mul_563);  view_249 = mul_563 = None
    add_286: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_560, mul_564);  mul_560 = mul_564 = None
    mul_565: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_801, add_286);  view_801 = add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_802: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_565, [1576, 1536]);  mul_565 = None
    mm_184: "f32[1576, 384]" = torch.ops.aten.mm.default(view_802, permute_601);  permute_601 = None
    permute_602: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_802, [1, 0])
    mm_185: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_602, view_248);  permute_602 = view_248 = None
    permute_603: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_205: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
    view_803: "f32[1536]" = torch.ops.aten.reshape.default(sum_205, [1536]);  sum_205 = None
    permute_604: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
    view_804: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_184, [8, 197, 384]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_567: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_804, primals_174);  primals_174 = None
    mul_568: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_567, 384)
    sum_206: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True)
    mul_569: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_567, mul_107);  mul_567 = None
    sum_207: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True);  mul_569 = None
    mul_570: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_107, sum_207);  sum_207 = None
    sub_193: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_568, sum_206);  mul_568 = sum_206 = None
    sub_194: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_193, mul_570);  sub_193 = mul_570 = None
    mul_571: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_55, sub_194);  div_55 = sub_194 = None
    mul_572: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_804, mul_107);  mul_107 = None
    sum_208: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 1]);  mul_572 = None
    sum_209: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_804, [0, 1]);  view_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_287: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_279, mul_571);  add_279 = mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_805: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_287, [1576, 384])
    mm_186: "f32[1576, 384]" = torch.ops.aten.mm.default(view_805, permute_605);  permute_605 = None
    permute_606: "f32[384, 1576]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_187: "f32[384, 384]" = torch.ops.aten.mm.default(permute_606, view_246);  permute_606 = view_246 = None
    permute_607: "f32[384, 384]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_210: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
    view_806: "f32[384]" = torch.ops.aten.reshape.default(sum_210, [384]);  sum_210 = None
    permute_608: "f32[384, 384]" = torch.ops.aten.permute.default(permute_607, [1, 0]);  permute_607 = None
    view_807: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_186, [8, 197, 384]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_808: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_807, [8, 197, 6, 64]);  view_807 = None
    permute_609: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None
    clone_245: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    view_809: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_245, [48, 197, 64]);  clone_245 = None
    bmm_96: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_610, view_809);  permute_610 = None
    bmm_97: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_809, permute_611);  view_809 = permute_611 = None
    view_810: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_96, [8, 6, 197, 64]);  bmm_96 = None
    view_811: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_97, [8, 6, 197, 197]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_573: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_811, alias_36);  view_811 = None
    sum_211: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [-1], True)
    mul_574: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_36, sum_211);  alias_36 = sum_211 = None
    sub_195: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_575: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_195, 0.125);  sub_195 = None
    view_812: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_575, [48, 197, 197]);  mul_575 = None
    bmm_98: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_612, view_812);  permute_612 = None
    bmm_99: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_812, permute_613);  view_812 = permute_613 = None
    view_813: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_98, [8, 6, 64, 197]);  bmm_98 = None
    view_814: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_99, [8, 6, 197, 64]);  bmm_99 = None
    permute_614: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_813, [0, 1, 3, 2]);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_615: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
    clone_246: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
    view_815: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_246, [8, 197, 384]);  clone_246 = None
    view_816: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_815, [1576, 384]);  view_815 = None
    permute_616: "f32[384, 1576]" = torch.ops.aten.permute.default(view_816, [1, 0])
    mm_188: "f32[384, 384]" = torch.ops.aten.mm.default(permute_616, view_233);  permute_616 = None
    permute_617: "f32[384, 384]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    mm_189: "f32[1576, 384]" = torch.ops.aten.mm.default(view_816, permute_618);  view_816 = permute_618 = None
    view_817: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_189, [8, 197, 384]);  mm_189 = None
    permute_619: "f32[384, 384]" = torch.ops.aten.permute.default(permute_617, [1, 0]);  permute_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_25: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_814, permute_614]);  view_814 = permute_614 = None
    view_818: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_25, [2, 8, 6, 197, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_620: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_818, [1, 3, 0, 2, 4]);  view_818 = None
    clone_247: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
    view_819: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_247, [8, 197, 768]);  clone_247 = None
    view_820: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_819, [1576, 768]);  view_819 = None
    permute_621: "f32[768, 1576]" = torch.ops.aten.permute.default(view_820, [1, 0])
    mm_190: "f32[768, 384]" = torch.ops.aten.mm.default(permute_621, view_233);  permute_621 = view_233 = None
    permute_622: "f32[384, 768]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    mm_191: "f32[1576, 384]" = torch.ops.aten.mm.default(view_820, permute_623);  view_820 = permute_623 = None
    view_821: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_191, [8, 197, 384]);  mm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_288: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_817, view_821);  view_817 = view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_624: "f32[768, 384]" = torch.ops.aten.permute.default(permute_622, [1, 0]);  permute_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_577: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_288, primals_168);  primals_168 = None
    mul_578: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_577, 384)
    sum_212: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2], True)
    mul_579: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_577, mul_104);  mul_577 = None
    sum_213: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True);  mul_579 = None
    mul_580: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_104, sum_213);  sum_213 = None
    sub_197: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_578, sum_212);  mul_578 = sum_212 = None
    sub_198: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_197, mul_580);  sub_197 = mul_580 = None
    div_56: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 384);  rsqrt_30 = None
    mul_581: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_56, sub_198);  div_56 = sub_198 = None
    mul_582: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_288, mul_104);  mul_104 = None
    sum_214: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 1]);  mul_582 = None
    sum_215: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_288, [0, 1]);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_289: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_287, mul_581);  add_287 = mul_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_64: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_289, 1, 0, 1)
    slice_65: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_289, 1, 1, 197);  add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_248: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_65, memory_format = torch.contiguous_format)
    view_822: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_248, [1568, 384]);  clone_248 = None
    mm_192: "f32[1568, 384]" = torch.ops.aten.mm.default(view_822, permute_625);  permute_625 = None
    permute_626: "f32[384, 1568]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_193: "f32[384, 384]" = torch.ops.aten.mm.default(permute_626, view_231);  permute_626 = view_231 = None
    permute_627: "f32[384, 384]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_216: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[384]" = torch.ops.aten.reshape.default(sum_216, [384]);  sum_216 = None
    permute_628: "f32[384, 384]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_824: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_192, [8, 196, 384]);  mm_192 = None
    view_825: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_824, [1568, 16, 24]);  view_824 = None
    mul_584: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_825, primals_164);  primals_164 = None
    mul_585: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_584, 24)
    sum_217: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True)
    mul_586: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_584, mul_102);  mul_584 = None
    sum_218: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True);  mul_586 = None
    mul_587: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_102, sum_218);  sum_218 = None
    sub_200: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_585, sum_217);  mul_585 = sum_217 = None
    sub_201: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_200, mul_587);  sub_200 = mul_587 = None
    mul_588: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_54, sub_201);  div_54 = sub_201 = None
    mul_589: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_825, mul_102);  mul_102 = None
    sum_219: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 1]);  mul_589 = None
    sum_220: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_825, [0, 1]);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_290: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_284, mul_588);  add_284 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_25: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_65, 1, 1, 9223372036854775807);  slice_65 = None
    slice_scatter_27: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_64, 1, 0, 1);  slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_291: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_25, slice_scatter_27);  slice_scatter_25 = slice_scatter_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_826: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_290, [25088, 24])
    mm_194: "f32[25088, 96]" = torch.ops.aten.mm.default(view_826, permute_629);  permute_629 = None
    permute_630: "f32[24, 25088]" = torch.ops.aten.permute.default(view_826, [1, 0])
    mm_195: "f32[24, 96]" = torch.ops.aten.mm.default(permute_630, view_228);  permute_630 = view_228 = None
    permute_631: "f32[96, 24]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_221: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_826, [0], True);  view_826 = None
    view_827: "f32[24]" = torch.ops.aten.reshape.default(sum_221, [24]);  sum_221 = None
    permute_632: "f32[24, 96]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_828: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_194, [1568, 16, 96]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_591: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_592: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, view_227)
    mul_593: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_592, -0.5);  mul_592 = None
    exp_37: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_593);  mul_593 = None
    mul_594: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_595: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, mul_594);  view_227 = mul_594 = None
    add_293: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_591, mul_595);  mul_591 = mul_595 = None
    mul_596: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_828, add_293);  view_828 = add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_829: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_596, [25088, 96]);  mul_596 = None
    mm_196: "f32[25088, 24]" = torch.ops.aten.mm.default(view_829, permute_633);  permute_633 = None
    permute_634: "f32[96, 25088]" = torch.ops.aten.permute.default(view_829, [1, 0])
    mm_197: "f32[96, 24]" = torch.ops.aten.mm.default(permute_634, view_226);  permute_634 = view_226 = None
    permute_635: "f32[24, 96]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_222: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_829, [0], True);  view_829 = None
    view_830: "f32[96]" = torch.ops.aten.reshape.default(sum_222, [96]);  sum_222 = None
    permute_636: "f32[96, 24]" = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
    view_831: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_196, [1568, 16, 24]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_598: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_831, primals_158);  primals_158 = None
    mul_599: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_598, 24)
    sum_223: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True)
    mul_600: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_598, mul_97);  mul_598 = None
    sum_224: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True);  mul_600 = None
    mul_601: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_97, sum_224);  sum_224 = None
    sub_203: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_599, sum_223);  mul_599 = sum_223 = None
    sub_204: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_203, mul_601);  sub_203 = mul_601 = None
    mul_602: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_58, sub_204);  div_58 = sub_204 = None
    mul_603: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_831, mul_97);  mul_97 = None
    sum_225: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 1]);  mul_603 = None
    sum_226: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_831, [0, 1]);  view_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_294: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_290, mul_602);  add_290 = mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_832: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_294, [25088, 24])
    mm_198: "f32[25088, 24]" = torch.ops.aten.mm.default(view_832, permute_637);  permute_637 = None
    permute_638: "f32[24, 25088]" = torch.ops.aten.permute.default(view_832, [1, 0])
    mm_199: "f32[24, 24]" = torch.ops.aten.mm.default(permute_638, view_224);  permute_638 = view_224 = None
    permute_639: "f32[24, 24]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_227: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_832, [0], True);  view_832 = None
    view_833: "f32[24]" = torch.ops.aten.reshape.default(sum_227, [24]);  sum_227 = None
    permute_640: "f32[24, 24]" = torch.ops.aten.permute.default(permute_639, [1, 0]);  permute_639 = None
    view_834: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_198, [1568, 16, 24]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_835: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_834, [1568, 16, 4, 6]);  view_834 = None
    permute_641: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_835, [0, 2, 1, 3]);  view_835 = None
    clone_251: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_641, memory_format = torch.contiguous_format);  permute_641 = None
    view_836: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_251, [6272, 16, 6]);  clone_251 = None
    bmm_100: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_642, view_836);  permute_642 = None
    bmm_101: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_836, permute_643);  view_836 = permute_643 = None
    view_837: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_100, [1568, 4, 16, 6]);  bmm_100 = None
    view_838: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_101, [1568, 4, 16, 16]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_604: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_838, alias_37);  view_838 = None
    sum_228: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_604, [-1], True)
    mul_605: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_37, sum_228);  alias_37 = sum_228 = None
    sub_205: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_606: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_205, 0.408248290463863);  sub_205 = None
    view_839: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_606, [6272, 16, 16]);  mul_606 = None
    bmm_102: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_644, view_839);  permute_644 = None
    bmm_103: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_839, permute_645);  view_839 = permute_645 = None
    view_840: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_102, [1568, 4, 6, 16]);  bmm_102 = None
    view_841: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_103, [1568, 4, 16, 6]);  bmm_103 = None
    permute_646: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_840, [0, 1, 3, 2]);  view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_647: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
    clone_252: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_647, memory_format = torch.contiguous_format);  permute_647 = None
    view_842: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_252, [1568, 16, 24]);  clone_252 = None
    view_843: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_842, [25088, 24]);  view_842 = None
    permute_648: "f32[24, 25088]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_200: "f32[24, 24]" = torch.ops.aten.mm.default(permute_648, view_211);  permute_648 = None
    permute_649: "f32[24, 24]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    mm_201: "f32[25088, 24]" = torch.ops.aten.mm.default(view_843, permute_650);  view_843 = permute_650 = None
    view_844: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_201, [1568, 16, 24]);  mm_201 = None
    permute_651: "f32[24, 24]" = torch.ops.aten.permute.default(permute_649, [1, 0]);  permute_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_26: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_841, permute_646]);  view_841 = permute_646 = None
    view_845: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_26, [2, 1568, 4, 16, 6]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_652: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_845, [1, 3, 0, 2, 4]);  view_845 = None
    clone_253: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
    view_846: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_253, [1568, 16, 48]);  clone_253 = None
    view_847: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_846, [25088, 48]);  view_846 = None
    permute_653: "f32[48, 25088]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_202: "f32[48, 24]" = torch.ops.aten.mm.default(permute_653, view_211);  permute_653 = view_211 = None
    permute_654: "f32[24, 48]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    mm_203: "f32[25088, 24]" = torch.ops.aten.mm.default(view_847, permute_655);  view_847 = permute_655 = None
    view_848: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_203, [1568, 16, 24]);  mm_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_295: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_844, view_848);  view_844 = view_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_656: "f32[48, 24]" = torch.ops.aten.permute.default(permute_654, [1, 0]);  permute_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_608: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_295, primals_152);  primals_152 = None
    mul_609: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_608, 24)
    sum_229: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True)
    mul_610: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_608, mul_84);  mul_608 = None
    sum_230: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [2], True);  mul_610 = None
    mul_611: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_84, sum_230);  sum_230 = None
    sub_207: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_609, sum_229);  mul_609 = sum_229 = None
    sub_208: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_207, mul_611);  sub_207 = mul_611 = None
    mul_612: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_59, sub_208);  sub_208 = None
    mul_613: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_295, mul_84)
    sum_231: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1]);  mul_613 = None
    sum_232: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_295, [0, 1]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_296: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_294, mul_612);  add_294 = mul_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_849: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_291, [1576, 384])
    mm_204: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_849, permute_657);  permute_657 = None
    permute_658: "f32[384, 1576]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_205: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_658, view_209);  permute_658 = view_209 = None
    permute_659: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_233: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[384]" = torch.ops.aten.reshape.default(sum_233, [384]);  sum_233 = None
    permute_660: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_659, [1, 0]);  permute_659 = None
    view_851: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_204, [8, 197, 1536]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_615: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_91, 0.5);  add_91 = None
    mul_616: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, view_208)
    mul_617: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_616, -0.5);  mul_616 = None
    exp_38: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_617);  mul_617 = None
    mul_618: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_619: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, mul_618);  view_208 = mul_618 = None
    add_298: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_615, mul_619);  mul_615 = mul_619 = None
    mul_620: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_851, add_298);  view_851 = add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_852: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_620, [1576, 1536]);  mul_620 = None
    mm_206: "f32[1576, 384]" = torch.ops.aten.mm.default(view_852, permute_661);  permute_661 = None
    permute_662: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_852, [1, 0])
    mm_207: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_662, view_207);  permute_662 = view_207 = None
    permute_663: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_234: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
    view_853: "f32[1536]" = torch.ops.aten.reshape.default(sum_234, [1536]);  sum_234 = None
    permute_664: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    view_854: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_206, [8, 197, 384]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_622: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_854, primals_146);  primals_146 = None
    mul_623: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_622, 384)
    sum_235: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True)
    mul_624: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_622, mul_89);  mul_622 = None
    sum_236: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True);  mul_624 = None
    mul_625: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_89, sum_236);  sum_236 = None
    sub_210: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_623, sum_235);  mul_623 = sum_235 = None
    sub_211: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_210, mul_625);  sub_210 = mul_625 = None
    mul_626: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_60, sub_211);  div_60 = sub_211 = None
    mul_627: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_854, mul_89);  mul_89 = None
    sum_237: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 1]);  mul_627 = None
    sum_238: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_299: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_291, mul_626);  add_291 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_855: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_299, [1576, 384])
    mm_208: "f32[1576, 384]" = torch.ops.aten.mm.default(view_855, permute_665);  permute_665 = None
    permute_666: "f32[384, 1576]" = torch.ops.aten.permute.default(view_855, [1, 0])
    mm_209: "f32[384, 384]" = torch.ops.aten.mm.default(permute_666, view_205);  permute_666 = view_205 = None
    permute_667: "f32[384, 384]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_239: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
    view_856: "f32[384]" = torch.ops.aten.reshape.default(sum_239, [384]);  sum_239 = None
    permute_668: "f32[384, 384]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_857: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_208, [8, 197, 384]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_858: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_857, [8, 197, 6, 64]);  view_857 = None
    permute_669: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
    clone_255: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
    view_859: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_255, [48, 197, 64]);  clone_255 = None
    bmm_104: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_670, view_859);  permute_670 = None
    bmm_105: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_859, permute_671);  view_859 = permute_671 = None
    view_860: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_104, [8, 6, 197, 64]);  bmm_104 = None
    view_861: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_105, [8, 6, 197, 197]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_628: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_861, alias_38);  view_861 = None
    sum_240: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [-1], True)
    mul_629: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_38, sum_240);  alias_38 = sum_240 = None
    sub_212: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_630: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_212, 0.125);  sub_212 = None
    view_862: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_630, [48, 197, 197]);  mul_630 = None
    bmm_106: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_672, view_862);  permute_672 = None
    bmm_107: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_862, permute_673);  view_862 = permute_673 = None
    view_863: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_106, [8, 6, 64, 197]);  bmm_106 = None
    view_864: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_107, [8, 6, 197, 64]);  bmm_107 = None
    permute_674: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_863, [0, 1, 3, 2]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_675: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
    clone_256: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_675, memory_format = torch.contiguous_format);  permute_675 = None
    view_865: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_256, [8, 197, 384]);  clone_256 = None
    view_866: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_865, [1576, 384]);  view_865 = None
    permute_676: "f32[384, 1576]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_210: "f32[384, 384]" = torch.ops.aten.mm.default(permute_676, view_192);  permute_676 = None
    permute_677: "f32[384, 384]" = torch.ops.aten.permute.default(mm_210, [1, 0]);  mm_210 = None
    mm_211: "f32[1576, 384]" = torch.ops.aten.mm.default(view_866, permute_678);  view_866 = permute_678 = None
    view_867: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_211, [8, 197, 384]);  mm_211 = None
    permute_679: "f32[384, 384]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_27: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_864, permute_674]);  view_864 = permute_674 = None
    view_868: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_27, [2, 8, 6, 197, 64]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_680: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_868, [1, 3, 0, 2, 4]);  view_868 = None
    clone_257: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_680, memory_format = torch.contiguous_format);  permute_680 = None
    view_869: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_257, [8, 197, 768]);  clone_257 = None
    view_870: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_869, [1576, 768]);  view_869 = None
    permute_681: "f32[768, 1576]" = torch.ops.aten.permute.default(view_870, [1, 0])
    mm_212: "f32[768, 384]" = torch.ops.aten.mm.default(permute_681, view_192);  permute_681 = view_192 = None
    permute_682: "f32[384, 768]" = torch.ops.aten.permute.default(mm_212, [1, 0]);  mm_212 = None
    mm_213: "f32[1576, 384]" = torch.ops.aten.mm.default(view_870, permute_683);  view_870 = permute_683 = None
    view_871: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_213, [8, 197, 384]);  mm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_300: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_867, view_871);  view_867 = view_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_684: "f32[768, 384]" = torch.ops.aten.permute.default(permute_682, [1, 0]);  permute_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_632: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_300, primals_140);  primals_140 = None
    mul_633: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_632, 384)
    sum_241: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [2], True)
    mul_634: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_632, mul_86);  mul_632 = None
    sum_242: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True);  mul_634 = None
    mul_635: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_86, sum_242);  sum_242 = None
    sub_214: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_633, sum_241);  mul_633 = sum_241 = None
    sub_215: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_214, mul_635);  sub_214 = mul_635 = None
    div_61: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 384);  rsqrt_25 = None
    mul_636: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_61, sub_215);  div_61 = sub_215 = None
    mul_637: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_300, mul_86);  mul_86 = None
    sum_243: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1]);  mul_637 = None
    sum_244: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_300, [0, 1]);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_301: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_299, mul_636);  add_299 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_66: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_301, 1, 0, 1)
    slice_67: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_301, 1, 1, 197);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_258: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_67, memory_format = torch.contiguous_format)
    view_872: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_258, [1568, 384]);  clone_258 = None
    mm_214: "f32[1568, 384]" = torch.ops.aten.mm.default(view_872, permute_685);  permute_685 = None
    permute_686: "f32[384, 1568]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_215: "f32[384, 384]" = torch.ops.aten.mm.default(permute_686, view_190);  permute_686 = view_190 = None
    permute_687: "f32[384, 384]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_245: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
    view_873: "f32[384]" = torch.ops.aten.reshape.default(sum_245, [384]);  sum_245 = None
    permute_688: "f32[384, 384]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    view_874: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_214, [8, 196, 384]);  mm_214 = None
    view_875: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_874, [1568, 16, 24]);  view_874 = None
    mul_639: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_875, primals_136);  primals_136 = None
    mul_640: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_639, 24)
    sum_246: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_639, [2], True)
    mul_641: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_639, mul_84);  mul_639 = None
    sum_247: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_641, [2], True);  mul_641 = None
    mul_642: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_84, sum_247);  sum_247 = None
    sub_217: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_640, sum_246);  mul_640 = sum_246 = None
    sub_218: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_217, mul_642);  sub_217 = mul_642 = None
    mul_643: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_59, sub_218);  div_59 = sub_218 = None
    mul_644: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_875, mul_84);  mul_84 = None
    sum_248: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 1]);  mul_644 = None
    sum_249: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_875, [0, 1]);  view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_302: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_296, mul_643);  add_296 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_29: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_67, 1, 1, 9223372036854775807);  slice_67 = None
    slice_scatter_31: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_66, 1, 0, 1);  slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_303: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_29, slice_scatter_31);  slice_scatter_29 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_876: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_302, [25088, 24])
    mm_216: "f32[25088, 96]" = torch.ops.aten.mm.default(view_876, permute_689);  permute_689 = None
    permute_690: "f32[24, 25088]" = torch.ops.aten.permute.default(view_876, [1, 0])
    mm_217: "f32[24, 96]" = torch.ops.aten.mm.default(permute_690, view_187);  permute_690 = view_187 = None
    permute_691: "f32[96, 24]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_250: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_876, [0], True);  view_876 = None
    view_877: "f32[24]" = torch.ops.aten.reshape.default(sum_250, [24]);  sum_250 = None
    permute_692: "f32[24, 96]" = torch.ops.aten.permute.default(permute_691, [1, 0]);  permute_691 = None
    view_878: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_216, [1568, 16, 96]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_646: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None
    mul_647: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, view_186)
    mul_648: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_647, -0.5);  mul_647 = None
    exp_39: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_648);  mul_648 = None
    mul_649: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_650: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, mul_649);  view_186 = mul_649 = None
    add_305: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_646, mul_650);  mul_646 = mul_650 = None
    mul_651: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_878, add_305);  view_878 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_879: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_651, [25088, 96]);  mul_651 = None
    mm_218: "f32[25088, 24]" = torch.ops.aten.mm.default(view_879, permute_693);  permute_693 = None
    permute_694: "f32[96, 25088]" = torch.ops.aten.permute.default(view_879, [1, 0])
    mm_219: "f32[96, 24]" = torch.ops.aten.mm.default(permute_694, view_185);  permute_694 = view_185 = None
    permute_695: "f32[24, 96]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_251: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_879, [0], True);  view_879 = None
    view_880: "f32[96]" = torch.ops.aten.reshape.default(sum_251, [96]);  sum_251 = None
    permute_696: "f32[96, 24]" = torch.ops.aten.permute.default(permute_695, [1, 0]);  permute_695 = None
    view_881: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_218, [1568, 16, 24]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_653: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_881, primals_130);  primals_130 = None
    mul_654: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_653, 24)
    sum_252: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True)
    mul_655: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_653, mul_79);  mul_653 = None
    sum_253: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
    mul_656: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_79, sum_253);  sum_253 = None
    sub_220: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_654, sum_252);  mul_654 = sum_252 = None
    sub_221: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_220, mul_656);  sub_220 = mul_656 = None
    mul_657: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_63, sub_221);  div_63 = sub_221 = None
    mul_658: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_881, mul_79);  mul_79 = None
    sum_254: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 1]);  mul_658 = None
    sum_255: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_881, [0, 1]);  view_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_306: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_302, mul_657);  add_302 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_882: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_306, [25088, 24])
    mm_220: "f32[25088, 24]" = torch.ops.aten.mm.default(view_882, permute_697);  permute_697 = None
    permute_698: "f32[24, 25088]" = torch.ops.aten.permute.default(view_882, [1, 0])
    mm_221: "f32[24, 24]" = torch.ops.aten.mm.default(permute_698, view_183);  permute_698 = view_183 = None
    permute_699: "f32[24, 24]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_256: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_882, [0], True);  view_882 = None
    view_883: "f32[24]" = torch.ops.aten.reshape.default(sum_256, [24]);  sum_256 = None
    permute_700: "f32[24, 24]" = torch.ops.aten.permute.default(permute_699, [1, 0]);  permute_699 = None
    view_884: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_220, [1568, 16, 24]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_885: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_884, [1568, 16, 4, 6]);  view_884 = None
    permute_701: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_885, [0, 2, 1, 3]);  view_885 = None
    clone_261: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_701, memory_format = torch.contiguous_format);  permute_701 = None
    view_886: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_261, [6272, 16, 6]);  clone_261 = None
    bmm_108: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_702, view_886);  permute_702 = None
    bmm_109: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_886, permute_703);  view_886 = permute_703 = None
    view_887: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_108, [1568, 4, 16, 6]);  bmm_108 = None
    view_888: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_109, [1568, 4, 16, 16]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_659: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_888, alias_39);  view_888 = None
    sum_257: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [-1], True)
    mul_660: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_39, sum_257);  alias_39 = sum_257 = None
    sub_222: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_661: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_222, 0.408248290463863);  sub_222 = None
    view_889: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_661, [6272, 16, 16]);  mul_661 = None
    bmm_110: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_704, view_889);  permute_704 = None
    bmm_111: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_889, permute_705);  view_889 = permute_705 = None
    view_890: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_110, [1568, 4, 6, 16]);  bmm_110 = None
    view_891: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_111, [1568, 4, 16, 6]);  bmm_111 = None
    permute_706: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_890, [0, 1, 3, 2]);  view_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_707: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_887, [0, 2, 1, 3]);  view_887 = None
    clone_262: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
    view_892: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_262, [1568, 16, 24]);  clone_262 = None
    view_893: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_892, [25088, 24]);  view_892 = None
    permute_708: "f32[24, 25088]" = torch.ops.aten.permute.default(view_893, [1, 0])
    mm_222: "f32[24, 24]" = torch.ops.aten.mm.default(permute_708, view_170);  permute_708 = None
    permute_709: "f32[24, 24]" = torch.ops.aten.permute.default(mm_222, [1, 0]);  mm_222 = None
    mm_223: "f32[25088, 24]" = torch.ops.aten.mm.default(view_893, permute_710);  view_893 = permute_710 = None
    view_894: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_223, [1568, 16, 24]);  mm_223 = None
    permute_711: "f32[24, 24]" = torch.ops.aten.permute.default(permute_709, [1, 0]);  permute_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_28: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_891, permute_706]);  view_891 = permute_706 = None
    view_895: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_28, [2, 1568, 4, 16, 6]);  cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_712: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_895, [1, 3, 0, 2, 4]);  view_895 = None
    clone_263: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
    view_896: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_263, [1568, 16, 48]);  clone_263 = None
    view_897: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_896, [25088, 48]);  view_896 = None
    permute_713: "f32[48, 25088]" = torch.ops.aten.permute.default(view_897, [1, 0])
    mm_224: "f32[48, 24]" = torch.ops.aten.mm.default(permute_713, view_170);  permute_713 = view_170 = None
    permute_714: "f32[24, 48]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    mm_225: "f32[25088, 24]" = torch.ops.aten.mm.default(view_897, permute_715);  view_897 = permute_715 = None
    view_898: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_225, [1568, 16, 24]);  mm_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_307: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_894, view_898);  view_894 = view_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_716: "f32[48, 24]" = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_663: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_307, primals_124);  primals_124 = None
    mul_664: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_663, 24)
    sum_258: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
    mul_665: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_663, mul_66);  mul_663 = None
    sum_259: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
    mul_666: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_66, sum_259);  sum_259 = None
    sub_224: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_664, sum_258);  mul_664 = sum_258 = None
    sub_225: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_224, mul_666);  sub_224 = mul_666 = None
    mul_667: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_64, sub_225);  sub_225 = None
    mul_668: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_307, mul_66)
    sum_260: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
    sum_261: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_307, [0, 1]);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_308: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_306, mul_667);  add_306 = mul_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_899: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_303, [1576, 384])
    mm_226: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_899, permute_717);  permute_717 = None
    permute_718: "f32[384, 1576]" = torch.ops.aten.permute.default(view_899, [1, 0])
    mm_227: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_718, view_168);  permute_718 = view_168 = None
    permute_719: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_262: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_899, [0], True);  view_899 = None
    view_900: "f32[384]" = torch.ops.aten.reshape.default(sum_262, [384]);  sum_262 = None
    permute_720: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
    view_901: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_226, [8, 197, 1536]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_670: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
    mul_671: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, view_167)
    mul_672: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_671, -0.5);  mul_671 = None
    exp_40: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_672);  mul_672 = None
    mul_673: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_674: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, mul_673);  view_167 = mul_673 = None
    add_310: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_670, mul_674);  mul_670 = mul_674 = None
    mul_675: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_901, add_310);  view_901 = add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_902: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_675, [1576, 1536]);  mul_675 = None
    mm_228: "f32[1576, 384]" = torch.ops.aten.mm.default(view_902, permute_721);  permute_721 = None
    permute_722: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_902, [1, 0])
    mm_229: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_722, view_166);  permute_722 = view_166 = None
    permute_723: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_263: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
    view_903: "f32[1536]" = torch.ops.aten.reshape.default(sum_263, [1536]);  sum_263 = None
    permute_724: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_904: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_228, [8, 197, 384]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_677: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_904, primals_118);  primals_118 = None
    mul_678: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_677, 384)
    sum_264: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True)
    mul_679: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_677, mul_71);  mul_677 = None
    sum_265: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True);  mul_679 = None
    mul_680: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_71, sum_265);  sum_265 = None
    sub_227: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_678, sum_264);  mul_678 = sum_264 = None
    sub_228: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_227, mul_680);  sub_227 = mul_680 = None
    mul_681: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_65, sub_228);  div_65 = sub_228 = None
    mul_682: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_904, mul_71);  mul_71 = None
    sum_266: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1]);  mul_682 = None
    sum_267: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_904, [0, 1]);  view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_311: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_303, mul_681);  add_303 = mul_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_905: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_311, [1576, 384])
    mm_230: "f32[1576, 384]" = torch.ops.aten.mm.default(view_905, permute_725);  permute_725 = None
    permute_726: "f32[384, 1576]" = torch.ops.aten.permute.default(view_905, [1, 0])
    mm_231: "f32[384, 384]" = torch.ops.aten.mm.default(permute_726, view_164);  permute_726 = view_164 = None
    permute_727: "f32[384, 384]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_268: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
    view_906: "f32[384]" = torch.ops.aten.reshape.default(sum_268, [384]);  sum_268 = None
    permute_728: "f32[384, 384]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_907: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_230, [8, 197, 384]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_908: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_907, [8, 197, 6, 64]);  view_907 = None
    permute_729: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_908, [0, 2, 1, 3]);  view_908 = None
    clone_265: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
    view_909: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_265, [48, 197, 64]);  clone_265 = None
    bmm_112: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_730, view_909);  permute_730 = None
    bmm_113: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_909, permute_731);  view_909 = permute_731 = None
    view_910: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_112, [8, 6, 197, 64]);  bmm_112 = None
    view_911: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_113, [8, 6, 197, 197]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_683: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_911, alias_40);  view_911 = None
    sum_269: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [-1], True)
    mul_684: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_40, sum_269);  alias_40 = sum_269 = None
    sub_229: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_685: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_229, 0.125);  sub_229 = None
    view_912: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_685, [48, 197, 197]);  mul_685 = None
    bmm_114: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_732, view_912);  permute_732 = None
    bmm_115: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_912, permute_733);  view_912 = permute_733 = None
    view_913: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_114, [8, 6, 64, 197]);  bmm_114 = None
    view_914: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_115, [8, 6, 197, 64]);  bmm_115 = None
    permute_734: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_913, [0, 1, 3, 2]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_735: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_910, [0, 2, 1, 3]);  view_910 = None
    clone_266: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_735, memory_format = torch.contiguous_format);  permute_735 = None
    view_915: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_266, [8, 197, 384]);  clone_266 = None
    view_916: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_915, [1576, 384]);  view_915 = None
    permute_736: "f32[384, 1576]" = torch.ops.aten.permute.default(view_916, [1, 0])
    mm_232: "f32[384, 384]" = torch.ops.aten.mm.default(permute_736, view_151);  permute_736 = None
    permute_737: "f32[384, 384]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    mm_233: "f32[1576, 384]" = torch.ops.aten.mm.default(view_916, permute_738);  view_916 = permute_738 = None
    view_917: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_233, [8, 197, 384]);  mm_233 = None
    permute_739: "f32[384, 384]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_29: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_914, permute_734]);  view_914 = permute_734 = None
    view_918: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_29, [2, 8, 6, 197, 64]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_740: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_918, [1, 3, 0, 2, 4]);  view_918 = None
    clone_267: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_740, memory_format = torch.contiguous_format);  permute_740 = None
    view_919: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_267, [8, 197, 768]);  clone_267 = None
    view_920: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_919, [1576, 768]);  view_919 = None
    permute_741: "f32[768, 1576]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_234: "f32[768, 384]" = torch.ops.aten.mm.default(permute_741, view_151);  permute_741 = view_151 = None
    permute_742: "f32[384, 768]" = torch.ops.aten.permute.default(mm_234, [1, 0]);  mm_234 = None
    mm_235: "f32[1576, 384]" = torch.ops.aten.mm.default(view_920, permute_743);  view_920 = permute_743 = None
    view_921: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_235, [8, 197, 384]);  mm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_312: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_917, view_921);  view_917 = view_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_744: "f32[768, 384]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_687: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_312, primals_112);  primals_112 = None
    mul_688: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_687, 384)
    sum_270: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True)
    mul_689: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_687, mul_68);  mul_687 = None
    sum_271: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True);  mul_689 = None
    mul_690: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_68, sum_271);  sum_271 = None
    sub_231: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_688, sum_270);  mul_688 = sum_270 = None
    sub_232: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_231, mul_690);  sub_231 = mul_690 = None
    div_66: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None
    mul_691: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_66, sub_232);  div_66 = sub_232 = None
    mul_692: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_312, mul_68);  mul_68 = None
    sum_272: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 1]);  mul_692 = None
    sum_273: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_312, [0, 1]);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_313: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_311, mul_691);  add_311 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_68: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_313, 1, 0, 1)
    slice_69: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_313, 1, 1, 197);  add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_268: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_69, memory_format = torch.contiguous_format)
    view_922: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_268, [1568, 384]);  clone_268 = None
    mm_236: "f32[1568, 384]" = torch.ops.aten.mm.default(view_922, permute_745);  permute_745 = None
    permute_746: "f32[384, 1568]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_237: "f32[384, 384]" = torch.ops.aten.mm.default(permute_746, view_149);  permute_746 = view_149 = None
    permute_747: "f32[384, 384]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_274: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[384]" = torch.ops.aten.reshape.default(sum_274, [384]);  sum_274 = None
    permute_748: "f32[384, 384]" = torch.ops.aten.permute.default(permute_747, [1, 0]);  permute_747 = None
    view_924: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_236, [8, 196, 384]);  mm_236 = None
    view_925: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_924, [1568, 16, 24]);  view_924 = None
    mul_694: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_925, primals_108);  primals_108 = None
    mul_695: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_694, 24)
    sum_275: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [2], True)
    mul_696: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_694, mul_66);  mul_694 = None
    sum_276: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True);  mul_696 = None
    mul_697: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_66, sum_276);  sum_276 = None
    sub_234: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_695, sum_275);  mul_695 = sum_275 = None
    sub_235: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_234, mul_697);  sub_234 = mul_697 = None
    mul_698: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_64, sub_235);  div_64 = sub_235 = None
    mul_699: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_925, mul_66);  mul_66 = None
    sum_277: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 1]);  mul_699 = None
    sum_278: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_925, [0, 1]);  view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_314: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_308, mul_698);  add_308 = mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_33: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_69, 1, 1, 9223372036854775807);  slice_69 = None
    slice_scatter_35: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_68, 1, 0, 1);  slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_315: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_33, slice_scatter_35);  slice_scatter_33 = slice_scatter_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_926: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_314, [25088, 24])
    mm_238: "f32[25088, 96]" = torch.ops.aten.mm.default(view_926, permute_749);  permute_749 = None
    permute_750: "f32[24, 25088]" = torch.ops.aten.permute.default(view_926, [1, 0])
    mm_239: "f32[24, 96]" = torch.ops.aten.mm.default(permute_750, view_146);  permute_750 = view_146 = None
    permute_751: "f32[96, 24]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_279: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True);  view_926 = None
    view_927: "f32[24]" = torch.ops.aten.reshape.default(sum_279, [24]);  sum_279 = None
    permute_752: "f32[24, 96]" = torch.ops.aten.permute.default(permute_751, [1, 0]);  permute_751 = None
    view_928: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_238, [1568, 16, 96]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_701: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_702: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, view_145)
    mul_703: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_702, -0.5);  mul_702 = None
    exp_41: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_703);  mul_703 = None
    mul_704: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_705: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, mul_704);  view_145 = mul_704 = None
    add_317: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_701, mul_705);  mul_701 = mul_705 = None
    mul_706: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_928, add_317);  view_928 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_929: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_706, [25088, 96]);  mul_706 = None
    mm_240: "f32[25088, 24]" = torch.ops.aten.mm.default(view_929, permute_753);  permute_753 = None
    permute_754: "f32[96, 25088]" = torch.ops.aten.permute.default(view_929, [1, 0])
    mm_241: "f32[96, 24]" = torch.ops.aten.mm.default(permute_754, view_144);  permute_754 = view_144 = None
    permute_755: "f32[24, 96]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_280: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_929, [0], True);  view_929 = None
    view_930: "f32[96]" = torch.ops.aten.reshape.default(sum_280, [96]);  sum_280 = None
    permute_756: "f32[96, 24]" = torch.ops.aten.permute.default(permute_755, [1, 0]);  permute_755 = None
    view_931: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_240, [1568, 16, 24]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_708: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_931, primals_102);  primals_102 = None
    mul_709: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_708, 24)
    sum_281: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [2], True)
    mul_710: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_708, mul_61);  mul_708 = None
    sum_282: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_710, [2], True);  mul_710 = None
    mul_711: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_61, sum_282);  sum_282 = None
    sub_237: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_709, sum_281);  mul_709 = sum_281 = None
    sub_238: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_237, mul_711);  sub_237 = mul_711 = None
    mul_712: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_68, sub_238);  div_68 = sub_238 = None
    mul_713: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_931, mul_61);  mul_61 = None
    sum_283: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 1]);  mul_713 = None
    sum_284: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_931, [0, 1]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_318: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_314, mul_712);  add_314 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_932: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_318, [25088, 24])
    mm_242: "f32[25088, 24]" = torch.ops.aten.mm.default(view_932, permute_757);  permute_757 = None
    permute_758: "f32[24, 25088]" = torch.ops.aten.permute.default(view_932, [1, 0])
    mm_243: "f32[24, 24]" = torch.ops.aten.mm.default(permute_758, view_142);  permute_758 = view_142 = None
    permute_759: "f32[24, 24]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_285: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_932, [0], True);  view_932 = None
    view_933: "f32[24]" = torch.ops.aten.reshape.default(sum_285, [24]);  sum_285 = None
    permute_760: "f32[24, 24]" = torch.ops.aten.permute.default(permute_759, [1, 0]);  permute_759 = None
    view_934: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_242, [1568, 16, 24]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_935: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_934, [1568, 16, 4, 6]);  view_934 = None
    permute_761: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_935, [0, 2, 1, 3]);  view_935 = None
    clone_271: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_761, memory_format = torch.contiguous_format);  permute_761 = None
    view_936: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_271, [6272, 16, 6]);  clone_271 = None
    bmm_116: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_762, view_936);  permute_762 = None
    bmm_117: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_936, permute_763);  view_936 = permute_763 = None
    view_937: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_116, [1568, 4, 16, 6]);  bmm_116 = None
    view_938: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_117, [1568, 4, 16, 16]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_714: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_938, alias_41);  view_938 = None
    sum_286: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [-1], True)
    mul_715: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_41, sum_286);  alias_41 = sum_286 = None
    sub_239: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_716: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_239, 0.408248290463863);  sub_239 = None
    view_939: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_716, [6272, 16, 16]);  mul_716 = None
    bmm_118: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_764, view_939);  permute_764 = None
    bmm_119: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_939, permute_765);  view_939 = permute_765 = None
    view_940: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_118, [1568, 4, 6, 16]);  bmm_118 = None
    view_941: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_119, [1568, 4, 16, 6]);  bmm_119 = None
    permute_766: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_940, [0, 1, 3, 2]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_767: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_937, [0, 2, 1, 3]);  view_937 = None
    clone_272: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_767, memory_format = torch.contiguous_format);  permute_767 = None
    view_942: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_272, [1568, 16, 24]);  clone_272 = None
    view_943: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_942, [25088, 24]);  view_942 = None
    permute_768: "f32[24, 25088]" = torch.ops.aten.permute.default(view_943, [1, 0])
    mm_244: "f32[24, 24]" = torch.ops.aten.mm.default(permute_768, view_129);  permute_768 = None
    permute_769: "f32[24, 24]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    mm_245: "f32[25088, 24]" = torch.ops.aten.mm.default(view_943, permute_770);  view_943 = permute_770 = None
    view_944: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_245, [1568, 16, 24]);  mm_245 = None
    permute_771: "f32[24, 24]" = torch.ops.aten.permute.default(permute_769, [1, 0]);  permute_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_30: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_941, permute_766]);  view_941 = permute_766 = None
    view_945: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_30, [2, 1568, 4, 16, 6]);  cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_772: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_945, [1, 3, 0, 2, 4]);  view_945 = None
    clone_273: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_772, memory_format = torch.contiguous_format);  permute_772 = None
    view_946: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_273, [1568, 16, 48]);  clone_273 = None
    view_947: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_946, [25088, 48]);  view_946 = None
    permute_773: "f32[48, 25088]" = torch.ops.aten.permute.default(view_947, [1, 0])
    mm_246: "f32[48, 24]" = torch.ops.aten.mm.default(permute_773, view_129);  permute_773 = view_129 = None
    permute_774: "f32[24, 48]" = torch.ops.aten.permute.default(mm_246, [1, 0]);  mm_246 = None
    mm_247: "f32[25088, 24]" = torch.ops.aten.mm.default(view_947, permute_775);  view_947 = permute_775 = None
    view_948: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_247, [1568, 16, 24]);  mm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_319: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_944, view_948);  view_944 = view_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_776: "f32[48, 24]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_718: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_319, primals_96);  primals_96 = None
    mul_719: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_718, 24)
    sum_287: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True)
    mul_720: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_718, mul_48);  mul_718 = None
    sum_288: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [2], True);  mul_720 = None
    mul_721: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_48, sum_288);  sum_288 = None
    sub_241: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_719, sum_287);  mul_719 = sum_287 = None
    sub_242: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_241, mul_721);  sub_241 = mul_721 = None
    mul_722: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_69, sub_242);  sub_242 = None
    mul_723: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_319, mul_48)
    sum_289: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_723, [0, 1]);  mul_723 = None
    sum_290: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_319, [0, 1]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_320: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_318, mul_722);  add_318 = mul_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_949: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_315, [1576, 384])
    mm_248: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_949, permute_777);  permute_777 = None
    permute_778: "f32[384, 1576]" = torch.ops.aten.permute.default(view_949, [1, 0])
    mm_249: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_778, view_127);  permute_778 = view_127 = None
    permute_779: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_291: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_949, [0], True);  view_949 = None
    view_950: "f32[384]" = torch.ops.aten.reshape.default(sum_291, [384]);  sum_291 = None
    permute_780: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
    view_951: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_248, [8, 197, 1536]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_725: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
    mul_726: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, view_126)
    mul_727: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_726, -0.5);  mul_726 = None
    exp_42: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_727);  mul_727 = None
    mul_728: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_729: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, mul_728);  view_126 = mul_728 = None
    add_322: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_725, mul_729);  mul_725 = mul_729 = None
    mul_730: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_951, add_322);  view_951 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_952: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_730, [1576, 1536]);  mul_730 = None
    mm_250: "f32[1576, 384]" = torch.ops.aten.mm.default(view_952, permute_781);  permute_781 = None
    permute_782: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_952, [1, 0])
    mm_251: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_782, view_125);  permute_782 = view_125 = None
    permute_783: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_292: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True);  view_952 = None
    view_953: "f32[1536]" = torch.ops.aten.reshape.default(sum_292, [1536]);  sum_292 = None
    permute_784: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_783, [1, 0]);  permute_783 = None
    view_954: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_250, [8, 197, 384]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_732: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_954, primals_90);  primals_90 = None
    mul_733: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_732, 384)
    sum_293: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_732, [2], True)
    mul_734: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_732, mul_53);  mul_732 = None
    sum_294: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_734, [2], True);  mul_734 = None
    mul_735: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_53, sum_294);  sum_294 = None
    sub_244: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_733, sum_293);  mul_733 = sum_293 = None
    sub_245: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_244, mul_735);  sub_244 = mul_735 = None
    mul_736: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_70, sub_245);  div_70 = sub_245 = None
    mul_737: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_954, mul_53);  mul_53 = None
    sum_295: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 1]);  mul_737 = None
    sum_296: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_954, [0, 1]);  view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_323: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_315, mul_736);  add_315 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_955: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_323, [1576, 384])
    mm_252: "f32[1576, 384]" = torch.ops.aten.mm.default(view_955, permute_785);  permute_785 = None
    permute_786: "f32[384, 1576]" = torch.ops.aten.permute.default(view_955, [1, 0])
    mm_253: "f32[384, 384]" = torch.ops.aten.mm.default(permute_786, view_123);  permute_786 = view_123 = None
    permute_787: "f32[384, 384]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_297: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_955, [0], True);  view_955 = None
    view_956: "f32[384]" = torch.ops.aten.reshape.default(sum_297, [384]);  sum_297 = None
    permute_788: "f32[384, 384]" = torch.ops.aten.permute.default(permute_787, [1, 0]);  permute_787 = None
    view_957: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_252, [8, 197, 384]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_958: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_957, [8, 197, 6, 64]);  view_957 = None
    permute_789: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_958, [0, 2, 1, 3]);  view_958 = None
    clone_275: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_789, memory_format = torch.contiguous_format);  permute_789 = None
    view_959: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_275, [48, 197, 64]);  clone_275 = None
    bmm_120: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_790, view_959);  permute_790 = None
    bmm_121: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_959, permute_791);  view_959 = permute_791 = None
    view_960: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_120, [8, 6, 197, 64]);  bmm_120 = None
    view_961: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_121, [8, 6, 197, 197]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_738: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_961, alias_42);  view_961 = None
    sum_298: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [-1], True)
    mul_739: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_42, sum_298);  alias_42 = sum_298 = None
    sub_246: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_738, mul_739);  mul_738 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_740: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_246, 0.125);  sub_246 = None
    view_962: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_740, [48, 197, 197]);  mul_740 = None
    bmm_122: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_792, view_962);  permute_792 = None
    bmm_123: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_962, permute_793);  view_962 = permute_793 = None
    view_963: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_122, [8, 6, 64, 197]);  bmm_122 = None
    view_964: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_123, [8, 6, 197, 64]);  bmm_123 = None
    permute_794: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_963, [0, 1, 3, 2]);  view_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_795: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None
    clone_276: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
    view_965: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_276, [8, 197, 384]);  clone_276 = None
    view_966: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_965, [1576, 384]);  view_965 = None
    permute_796: "f32[384, 1576]" = torch.ops.aten.permute.default(view_966, [1, 0])
    mm_254: "f32[384, 384]" = torch.ops.aten.mm.default(permute_796, view_110);  permute_796 = None
    permute_797: "f32[384, 384]" = torch.ops.aten.permute.default(mm_254, [1, 0]);  mm_254 = None
    mm_255: "f32[1576, 384]" = torch.ops.aten.mm.default(view_966, permute_798);  view_966 = permute_798 = None
    view_967: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_255, [8, 197, 384]);  mm_255 = None
    permute_799: "f32[384, 384]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_31: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_964, permute_794]);  view_964 = permute_794 = None
    view_968: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_31, [2, 8, 6, 197, 64]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_800: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_968, [1, 3, 0, 2, 4]);  view_968 = None
    clone_277: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_800, memory_format = torch.contiguous_format);  permute_800 = None
    view_969: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_277, [8, 197, 768]);  clone_277 = None
    view_970: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_969, [1576, 768]);  view_969 = None
    permute_801: "f32[768, 1576]" = torch.ops.aten.permute.default(view_970, [1, 0])
    mm_256: "f32[768, 384]" = torch.ops.aten.mm.default(permute_801, view_110);  permute_801 = view_110 = None
    permute_802: "f32[384, 768]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    mm_257: "f32[1576, 384]" = torch.ops.aten.mm.default(view_970, permute_803);  view_970 = permute_803 = None
    view_971: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_257, [8, 197, 384]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_324: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_967, view_971);  view_967 = view_971 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_804: "f32[768, 384]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_742: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_324, primals_84);  primals_84 = None
    mul_743: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_742, 384)
    sum_299: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [2], True)
    mul_744: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_742, mul_50);  mul_742 = None
    sum_300: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True);  mul_744 = None
    mul_745: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_50, sum_300);  sum_300 = None
    sub_248: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_743, sum_299);  mul_743 = sum_299 = None
    sub_249: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_248, mul_745);  sub_248 = mul_745 = None
    div_71: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None
    mul_746: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_71, sub_249);  div_71 = sub_249 = None
    mul_747: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_324, mul_50);  mul_50 = None
    sum_301: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 1]);  mul_747 = None
    sum_302: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_324, [0, 1]);  add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_325: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_323, mul_746);  add_323 = mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_70: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_325, 1, 0, 1)
    slice_71: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_325, 1, 1, 197);  add_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_278: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_71, memory_format = torch.contiguous_format)
    view_972: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_278, [1568, 384]);  clone_278 = None
    mm_258: "f32[1568, 384]" = torch.ops.aten.mm.default(view_972, permute_805);  permute_805 = None
    permute_806: "f32[384, 1568]" = torch.ops.aten.permute.default(view_972, [1, 0])
    mm_259: "f32[384, 384]" = torch.ops.aten.mm.default(permute_806, view_108);  permute_806 = view_108 = None
    permute_807: "f32[384, 384]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_303: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_972, [0], True);  view_972 = None
    view_973: "f32[384]" = torch.ops.aten.reshape.default(sum_303, [384]);  sum_303 = None
    permute_808: "f32[384, 384]" = torch.ops.aten.permute.default(permute_807, [1, 0]);  permute_807 = None
    view_974: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_258, [8, 196, 384]);  mm_258 = None
    view_975: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_974, [1568, 16, 24]);  view_974 = None
    mul_749: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_975, primals_80);  primals_80 = None
    mul_750: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_749, 24)
    sum_304: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
    mul_751: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_749, mul_48);  mul_749 = None
    sum_305: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
    mul_752: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_48, sum_305);  sum_305 = None
    sub_251: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_750, sum_304);  mul_750 = sum_304 = None
    sub_252: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_251, mul_752);  sub_251 = mul_752 = None
    mul_753: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_69, sub_252);  div_69 = sub_252 = None
    mul_754: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_975, mul_48);  mul_48 = None
    sum_306: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
    sum_307: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_975, [0, 1]);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_326: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_320, mul_753);  add_320 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_37: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_71, 1, 1, 9223372036854775807);  slice_71 = None
    slice_scatter_39: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_70, 1, 0, 1);  slice_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_327: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_37, slice_scatter_39);  slice_scatter_37 = slice_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_976: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_326, [25088, 24])
    mm_260: "f32[25088, 96]" = torch.ops.aten.mm.default(view_976, permute_809);  permute_809 = None
    permute_810: "f32[24, 25088]" = torch.ops.aten.permute.default(view_976, [1, 0])
    mm_261: "f32[24, 96]" = torch.ops.aten.mm.default(permute_810, view_105);  permute_810 = view_105 = None
    permute_811: "f32[96, 24]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_308: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
    view_977: "f32[24]" = torch.ops.aten.reshape.default(sum_308, [24]);  sum_308 = None
    permute_812: "f32[24, 96]" = torch.ops.aten.permute.default(permute_811, [1, 0]);  permute_811 = None
    view_978: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_260, [1568, 16, 96]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_756: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None
    mul_757: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, view_104)
    mul_758: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_757, -0.5);  mul_757 = None
    exp_43: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_758);  mul_758 = None
    mul_759: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_760: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, mul_759);  view_104 = mul_759 = None
    add_329: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_756, mul_760);  mul_756 = mul_760 = None
    mul_761: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_978, add_329);  view_978 = add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_979: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_761, [25088, 96]);  mul_761 = None
    mm_262: "f32[25088, 24]" = torch.ops.aten.mm.default(view_979, permute_813);  permute_813 = None
    permute_814: "f32[96, 25088]" = torch.ops.aten.permute.default(view_979, [1, 0])
    mm_263: "f32[96, 24]" = torch.ops.aten.mm.default(permute_814, view_103);  permute_814 = view_103 = None
    permute_815: "f32[24, 96]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_309: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
    view_980: "f32[96]" = torch.ops.aten.reshape.default(sum_309, [96]);  sum_309 = None
    permute_816: "f32[96, 24]" = torch.ops.aten.permute.default(permute_815, [1, 0]);  permute_815 = None
    view_981: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_262, [1568, 16, 24]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_763: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_981, primals_74);  primals_74 = None
    mul_764: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_763, 24)
    sum_310: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True)
    mul_765: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_763, mul_43);  mul_763 = None
    sum_311: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True);  mul_765 = None
    mul_766: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_43, sum_311);  sum_311 = None
    sub_254: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_764, sum_310);  mul_764 = sum_310 = None
    sub_255: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_254, mul_766);  sub_254 = mul_766 = None
    mul_767: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_73, sub_255);  div_73 = sub_255 = None
    mul_768: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_981, mul_43);  mul_43 = None
    sum_312: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 1]);  mul_768 = None
    sum_313: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_981, [0, 1]);  view_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_330: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_326, mul_767);  add_326 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_982: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_330, [25088, 24])
    mm_264: "f32[25088, 24]" = torch.ops.aten.mm.default(view_982, permute_817);  permute_817 = None
    permute_818: "f32[24, 25088]" = torch.ops.aten.permute.default(view_982, [1, 0])
    mm_265: "f32[24, 24]" = torch.ops.aten.mm.default(permute_818, view_101);  permute_818 = view_101 = None
    permute_819: "f32[24, 24]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_314: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
    view_983: "f32[24]" = torch.ops.aten.reshape.default(sum_314, [24]);  sum_314 = None
    permute_820: "f32[24, 24]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
    view_984: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_264, [1568, 16, 24]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_985: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_984, [1568, 16, 4, 6]);  view_984 = None
    permute_821: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None
    clone_281: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
    view_986: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_281, [6272, 16, 6]);  clone_281 = None
    bmm_124: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_822, view_986);  permute_822 = None
    bmm_125: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_986, permute_823);  view_986 = permute_823 = None
    view_987: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_124, [1568, 4, 16, 6]);  bmm_124 = None
    view_988: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_125, [1568, 4, 16, 16]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_769: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_988, alias_43);  view_988 = None
    sum_315: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [-1], True)
    mul_770: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_43, sum_315);  alias_43 = sum_315 = None
    sub_256: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_771: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_256, 0.408248290463863);  sub_256 = None
    view_989: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_771, [6272, 16, 16]);  mul_771 = None
    bmm_126: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_824, view_989);  permute_824 = None
    bmm_127: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_989, permute_825);  view_989 = permute_825 = None
    view_990: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_126, [1568, 4, 6, 16]);  bmm_126 = None
    view_991: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_127, [1568, 4, 16, 6]);  bmm_127 = None
    permute_826: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_990, [0, 1, 3, 2]);  view_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_827: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_987, [0, 2, 1, 3]);  view_987 = None
    clone_282: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_827, memory_format = torch.contiguous_format);  permute_827 = None
    view_992: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_282, [1568, 16, 24]);  clone_282 = None
    view_993: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_992, [25088, 24]);  view_992 = None
    permute_828: "f32[24, 25088]" = torch.ops.aten.permute.default(view_993, [1, 0])
    mm_266: "f32[24, 24]" = torch.ops.aten.mm.default(permute_828, view_88);  permute_828 = None
    permute_829: "f32[24, 24]" = torch.ops.aten.permute.default(mm_266, [1, 0]);  mm_266 = None
    mm_267: "f32[25088, 24]" = torch.ops.aten.mm.default(view_993, permute_830);  view_993 = permute_830 = None
    view_994: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_267, [1568, 16, 24]);  mm_267 = None
    permute_831: "f32[24, 24]" = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_32: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_991, permute_826]);  view_991 = permute_826 = None
    view_995: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_32, [2, 1568, 4, 16, 6]);  cat_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_832: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_995, [1, 3, 0, 2, 4]);  view_995 = None
    clone_283: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_832, memory_format = torch.contiguous_format);  permute_832 = None
    view_996: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_283, [1568, 16, 48]);  clone_283 = None
    view_997: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_996, [25088, 48]);  view_996 = None
    permute_833: "f32[48, 25088]" = torch.ops.aten.permute.default(view_997, [1, 0])
    mm_268: "f32[48, 24]" = torch.ops.aten.mm.default(permute_833, view_88);  permute_833 = view_88 = None
    permute_834: "f32[24, 48]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    mm_269: "f32[25088, 24]" = torch.ops.aten.mm.default(view_997, permute_835);  view_997 = permute_835 = None
    view_998: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_269, [1568, 16, 24]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_331: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_994, view_998);  view_994 = view_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_836: "f32[48, 24]" = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_773: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_331, primals_68);  primals_68 = None
    mul_774: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_773, 24)
    sum_316: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True)
    mul_775: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_773, mul_30);  mul_773 = None
    sum_317: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
    mul_776: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_30, sum_317);  sum_317 = None
    sub_258: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_774, sum_316);  mul_774 = sum_316 = None
    sub_259: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_258, mul_776);  sub_258 = mul_776 = None
    mul_777: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_74, sub_259);  sub_259 = None
    mul_778: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_331, mul_30)
    sum_318: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 1]);  mul_778 = None
    sum_319: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_331, [0, 1]);  add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_332: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_330, mul_777);  add_330 = mul_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_999: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_327, [1576, 384])
    mm_270: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_999, permute_837);  permute_837 = None
    permute_838: "f32[384, 1576]" = torch.ops.aten.permute.default(view_999, [1, 0])
    mm_271: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_838, view_86);  permute_838 = view_86 = None
    permute_839: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_320: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_999, [0], True);  view_999 = None
    view_1000: "f32[384]" = torch.ops.aten.reshape.default(sum_320, [384]);  sum_320 = None
    permute_840: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_839, [1, 0]);  permute_839 = None
    view_1001: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_270, [8, 197, 1536]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_780: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_781: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_782: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_781, -0.5);  mul_781 = None
    exp_44: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_782);  mul_782 = None
    mul_783: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_784: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, mul_783);  view_85 = mul_783 = None
    add_334: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_780, mul_784);  mul_780 = mul_784 = None
    mul_785: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_1001, add_334);  view_1001 = add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1002: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_785, [1576, 1536]);  mul_785 = None
    mm_272: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1002, permute_841);  permute_841 = None
    permute_842: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_1002, [1, 0])
    mm_273: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_842, view_84);  permute_842 = view_84 = None
    permute_843: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_321: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
    view_1003: "f32[1536]" = torch.ops.aten.reshape.default(sum_321, [1536]);  sum_321 = None
    permute_844: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
    view_1004: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_272, [8, 197, 384]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_787: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_1004, primals_62);  primals_62 = None
    mul_788: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_787, 384)
    sum_322: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_787, [2], True)
    mul_789: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_787, mul_35);  mul_787 = None
    sum_323: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True);  mul_789 = None
    mul_790: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_35, sum_323);  sum_323 = None
    sub_261: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_788, sum_322);  mul_788 = sum_322 = None
    sub_262: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_261, mul_790);  sub_261 = mul_790 = None
    mul_791: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_75, sub_262);  div_75 = sub_262 = None
    mul_792: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_1004, mul_35);  mul_35 = None
    sum_324: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 1]);  mul_792 = None
    sum_325: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_1004, [0, 1]);  view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_335: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_327, mul_791);  add_327 = mul_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_1005: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_335, [1576, 384])
    mm_274: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1005, permute_845);  permute_845 = None
    permute_846: "f32[384, 1576]" = torch.ops.aten.permute.default(view_1005, [1, 0])
    mm_275: "f32[384, 384]" = torch.ops.aten.mm.default(permute_846, view_82);  permute_846 = view_82 = None
    permute_847: "f32[384, 384]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_326: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
    view_1006: "f32[384]" = torch.ops.aten.reshape.default(sum_326, [384]);  sum_326 = None
    permute_848: "f32[384, 384]" = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
    view_1007: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_274, [8, 197, 384]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_1008: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_1007, [8, 197, 6, 64]);  view_1007 = None
    permute_849: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_1008, [0, 2, 1, 3]);  view_1008 = None
    clone_285: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_849, memory_format = torch.contiguous_format);  permute_849 = None
    view_1009: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_285, [48, 197, 64]);  clone_285 = None
    bmm_128: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_850, view_1009);  permute_850 = None
    bmm_129: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_1009, permute_851);  view_1009 = permute_851 = None
    view_1010: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_128, [8, 6, 197, 64]);  bmm_128 = None
    view_1011: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_129, [8, 6, 197, 197]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_793: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_1011, alias_44);  view_1011 = None
    sum_327: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_793, [-1], True)
    mul_794: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_44, sum_327);  alias_44 = sum_327 = None
    sub_263: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_795: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_263, 0.125);  sub_263 = None
    view_1012: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_795, [48, 197, 197]);  mul_795 = None
    bmm_130: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_852, view_1012);  permute_852 = None
    bmm_131: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_1012, permute_853);  view_1012 = permute_853 = None
    view_1013: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_130, [8, 6, 64, 197]);  bmm_130 = None
    view_1014: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_131, [8, 6, 197, 64]);  bmm_131 = None
    permute_854: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_1013, [0, 1, 3, 2]);  view_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_855: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_1010, [0, 2, 1, 3]);  view_1010 = None
    clone_286: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_855, memory_format = torch.contiguous_format);  permute_855 = None
    view_1015: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_286, [8, 197, 384]);  clone_286 = None
    view_1016: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_1015, [1576, 384]);  view_1015 = None
    permute_856: "f32[384, 1576]" = torch.ops.aten.permute.default(view_1016, [1, 0])
    mm_276: "f32[384, 384]" = torch.ops.aten.mm.default(permute_856, view_69);  permute_856 = None
    permute_857: "f32[384, 384]" = torch.ops.aten.permute.default(mm_276, [1, 0]);  mm_276 = None
    mm_277: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1016, permute_858);  view_1016 = permute_858 = None
    view_1017: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_277, [8, 197, 384]);  mm_277 = None
    permute_859: "f32[384, 384]" = torch.ops.aten.permute.default(permute_857, [1, 0]);  permute_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_33: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_1014, permute_854]);  view_1014 = permute_854 = None
    view_1018: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_33, [2, 8, 6, 197, 64]);  cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_860: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_1018, [1, 3, 0, 2, 4]);  view_1018 = None
    clone_287: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_860, memory_format = torch.contiguous_format);  permute_860 = None
    view_1019: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_287, [8, 197, 768]);  clone_287 = None
    view_1020: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_1019, [1576, 768]);  view_1019 = None
    permute_861: "f32[768, 1576]" = torch.ops.aten.permute.default(view_1020, [1, 0])
    mm_278: "f32[768, 384]" = torch.ops.aten.mm.default(permute_861, view_69);  permute_861 = view_69 = None
    permute_862: "f32[384, 768]" = torch.ops.aten.permute.default(mm_278, [1, 0]);  mm_278 = None
    mm_279: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1020, permute_863);  view_1020 = permute_863 = None
    view_1021: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_279, [8, 197, 384]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_336: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_1017, view_1021);  view_1017 = view_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_864: "f32[768, 384]" = torch.ops.aten.permute.default(permute_862, [1, 0]);  permute_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_797: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_336, primals_56);  primals_56 = None
    mul_798: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_797, 384)
    sum_328: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True)
    mul_799: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_797, mul_32);  mul_797 = None
    sum_329: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_799, [2], True);  mul_799 = None
    mul_800: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_32, sum_329);  sum_329 = None
    sub_265: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_798, sum_328);  mul_798 = sum_328 = None
    sub_266: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_265, mul_800);  sub_265 = mul_800 = None
    div_76: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 384);  rsqrt_10 = None
    mul_801: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_76, sub_266);  div_76 = sub_266 = None
    mul_802: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_336, mul_32);  mul_32 = None
    sum_330: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 1]);  mul_802 = None
    sum_331: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_336, [0, 1]);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_337: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_335, mul_801);  add_335 = mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_72: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_337, 1, 0, 1)
    slice_73: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_337, 1, 1, 197);  add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_288: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_73, memory_format = torch.contiguous_format)
    view_1022: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_288, [1568, 384]);  clone_288 = None
    mm_280: "f32[1568, 384]" = torch.ops.aten.mm.default(view_1022, permute_865);  permute_865 = None
    permute_866: "f32[384, 1568]" = torch.ops.aten.permute.default(view_1022, [1, 0])
    mm_281: "f32[384, 384]" = torch.ops.aten.mm.default(permute_866, view_67);  permute_866 = view_67 = None
    permute_867: "f32[384, 384]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_332: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1022, [0], True);  view_1022 = None
    view_1023: "f32[384]" = torch.ops.aten.reshape.default(sum_332, [384]);  sum_332 = None
    permute_868: "f32[384, 384]" = torch.ops.aten.permute.default(permute_867, [1, 0]);  permute_867 = None
    view_1024: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_280, [8, 196, 384]);  mm_280 = None
    view_1025: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_1024, [1568, 16, 24]);  view_1024 = None
    mul_804: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1025, primals_52);  primals_52 = None
    mul_805: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_804, 24)
    sum_333: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True)
    mul_806: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_804, mul_30);  mul_804 = None
    sum_334: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [2], True);  mul_806 = None
    mul_807: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_30, sum_334);  sum_334 = None
    sub_268: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_805, sum_333);  mul_805 = sum_333 = None
    sub_269: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_268, mul_807);  sub_268 = mul_807 = None
    mul_808: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_74, sub_269);  div_74 = sub_269 = None
    mul_809: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1025, mul_30);  mul_30 = None
    sum_335: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 1]);  mul_809 = None
    sum_336: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_1025, [0, 1]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_338: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_332, mul_808);  add_332 = mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_41: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_73, 1, 1, 9223372036854775807);  slice_73 = None
    slice_scatter_43: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_72, 1, 0, 1);  slice_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_339: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_41, slice_scatter_43);  slice_scatter_41 = slice_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1026: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_338, [25088, 24])
    mm_282: "f32[25088, 96]" = torch.ops.aten.mm.default(view_1026, permute_869);  permute_869 = None
    permute_870: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1026, [1, 0])
    mm_283: "f32[24, 96]" = torch.ops.aten.mm.default(permute_870, view_64);  permute_870 = view_64 = None
    permute_871: "f32[96, 24]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_337: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_1026, [0], True);  view_1026 = None
    view_1027: "f32[24]" = torch.ops.aten.reshape.default(sum_337, [24]);  sum_337 = None
    permute_872: "f32[24, 96]" = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
    view_1028: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_282, [1568, 16, 96]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_811: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_812: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_813: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_45: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_815: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, mul_814);  view_63 = mul_814 = None
    add_341: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_1028, add_341);  view_1028 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1029: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_816, [25088, 96]);  mul_816 = None
    mm_284: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1029, permute_873);  permute_873 = None
    permute_874: "f32[96, 25088]" = torch.ops.aten.permute.default(view_1029, [1, 0])
    mm_285: "f32[96, 24]" = torch.ops.aten.mm.default(permute_874, view_62);  permute_874 = view_62 = None
    permute_875: "f32[24, 96]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_338: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_1029, [0], True);  view_1029 = None
    view_1030: "f32[96]" = torch.ops.aten.reshape.default(sum_338, [96]);  sum_338 = None
    permute_876: "f32[96, 24]" = torch.ops.aten.permute.default(permute_875, [1, 0]);  permute_875 = None
    view_1031: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_284, [1568, 16, 24]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_818: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1031, primals_46);  primals_46 = None
    mul_819: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_818, 24)
    sum_339: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [2], True)
    mul_820: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_818, mul_25);  mul_818 = None
    sum_340: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
    mul_821: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_25, sum_340);  sum_340 = None
    sub_271: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_819, sum_339);  mul_819 = sum_339 = None
    sub_272: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_271, mul_821);  sub_271 = mul_821 = None
    mul_822: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_78, sub_272);  div_78 = sub_272 = None
    mul_823: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1031, mul_25);  mul_25 = None
    sum_341: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
    sum_342: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_1031, [0, 1]);  view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_342: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_338, mul_822);  add_338 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_1032: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_342, [25088, 24])
    mm_286: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1032, permute_877);  permute_877 = None
    permute_878: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1032, [1, 0])
    mm_287: "f32[24, 24]" = torch.ops.aten.mm.default(permute_878, view_60);  permute_878 = view_60 = None
    permute_879: "f32[24, 24]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    sum_343: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_1032, [0], True);  view_1032 = None
    view_1033: "f32[24]" = torch.ops.aten.reshape.default(sum_343, [24]);  sum_343 = None
    permute_880: "f32[24, 24]" = torch.ops.aten.permute.default(permute_879, [1, 0]);  permute_879 = None
    view_1034: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_286, [1568, 16, 24]);  mm_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_1035: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_1034, [1568, 16, 4, 6]);  view_1034 = None
    permute_881: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_1035, [0, 2, 1, 3]);  view_1035 = None
    clone_291: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_1036: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_291, [6272, 16, 6]);  clone_291 = None
    bmm_132: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_882, view_1036);  permute_882 = None
    bmm_133: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_1036, permute_883);  view_1036 = permute_883 = None
    view_1037: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_132, [1568, 4, 16, 6]);  bmm_132 = None
    view_1038: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_133, [1568, 4, 16, 16]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_824: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_1038, alias_45);  view_1038 = None
    sum_344: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [-1], True)
    mul_825: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_45, sum_344);  alias_45 = sum_344 = None
    sub_273: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_826: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_273, 0.408248290463863);  sub_273 = None
    view_1039: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_826, [6272, 16, 16]);  mul_826 = None
    bmm_134: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_884, view_1039);  permute_884 = None
    bmm_135: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_1039, permute_885);  view_1039 = permute_885 = None
    view_1040: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_134, [1568, 4, 6, 16]);  bmm_134 = None
    view_1041: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_135, [1568, 4, 16, 6]);  bmm_135 = None
    permute_886: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_1040, [0, 1, 3, 2]);  view_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_887: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_1037, [0, 2, 1, 3]);  view_1037 = None
    clone_292: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
    view_1042: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_292, [1568, 16, 24]);  clone_292 = None
    view_1043: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_1042, [25088, 24]);  view_1042 = None
    permute_888: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_288: "f32[24, 24]" = torch.ops.aten.mm.default(permute_888, view_47);  permute_888 = None
    permute_889: "f32[24, 24]" = torch.ops.aten.permute.default(mm_288, [1, 0]);  mm_288 = None
    mm_289: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1043, permute_890);  view_1043 = permute_890 = None
    view_1044: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_289, [1568, 16, 24]);  mm_289 = None
    permute_891: "f32[24, 24]" = torch.ops.aten.permute.default(permute_889, [1, 0]);  permute_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_34: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_1041, permute_886]);  view_1041 = permute_886 = None
    view_1045: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_34, [2, 1568, 4, 16, 6]);  cat_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_892: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_1045, [1, 3, 0, 2, 4]);  view_1045 = None
    clone_293: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_892, memory_format = torch.contiguous_format);  permute_892 = None
    view_1046: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_293, [1568, 16, 48]);  clone_293 = None
    view_1047: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_1046, [25088, 48]);  view_1046 = None
    permute_893: "f32[48, 25088]" = torch.ops.aten.permute.default(view_1047, [1, 0])
    mm_290: "f32[48, 24]" = torch.ops.aten.mm.default(permute_893, view_47);  permute_893 = view_47 = None
    permute_894: "f32[24, 48]" = torch.ops.aten.permute.default(mm_290, [1, 0]);  mm_290 = None
    mm_291: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1047, permute_895);  view_1047 = permute_895 = None
    view_1048: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_291, [1568, 16, 24]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_343: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_1044, view_1048);  view_1044 = view_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_896: "f32[48, 24]" = torch.ops.aten.permute.default(permute_894, [1, 0]);  permute_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_828: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_343, primals_40);  primals_40 = None
    mul_829: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_828, 24)
    sum_345: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_828, [2], True)
    mul_830: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_828, mul_12);  mul_828 = None
    sum_346: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_830, [2], True);  mul_830 = None
    mul_831: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_12, sum_346);  sum_346 = None
    sub_275: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_829, sum_345);  mul_829 = sum_345 = None
    sub_276: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_275, mul_831);  sub_275 = mul_831 = None
    mul_832: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_79, sub_276);  sub_276 = None
    mul_833: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_343, mul_12)
    sum_347: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 1]);  mul_833 = None
    sum_348: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 1]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_344: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_342, mul_832);  add_342 = mul_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1049: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_339, [1576, 384])
    mm_292: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_1049, permute_897);  permute_897 = None
    permute_898: "f32[384, 1576]" = torch.ops.aten.permute.default(view_1049, [1, 0])
    mm_293: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_898, view_45);  permute_898 = view_45 = None
    permute_899: "f32[1536, 384]" = torch.ops.aten.permute.default(mm_293, [1, 0]);  mm_293 = None
    sum_349: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1049, [0], True);  view_1049 = None
    view_1050: "f32[384]" = torch.ops.aten.reshape.default(sum_349, [384]);  sum_349 = None
    permute_900: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_899, [1, 0]);  permute_899 = None
    view_1051: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(mm_292, [8, 197, 1536]);  mm_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_835: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None
    mul_836: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, view_44)
    mul_837: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_836, -0.5);  mul_836 = None
    exp_46: "f32[8, 197, 1536]" = torch.ops.aten.exp.default(mul_837);  mul_837 = None
    mul_838: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_839: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, mul_838);  view_44 = mul_838 = None
    add_346: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(mul_835, mul_839);  mul_835 = mul_839 = None
    mul_840: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_1051, add_346);  view_1051 = add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1052: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_840, [1576, 1536]);  mul_840 = None
    mm_294: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1052, permute_901);  permute_901 = None
    permute_902: "f32[1536, 1576]" = torch.ops.aten.permute.default(view_1052, [1, 0])
    mm_295: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_902, view_43);  permute_902 = view_43 = None
    permute_903: "f32[384, 1536]" = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
    sum_350: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1052, [0], True);  view_1052 = None
    view_1053: "f32[1536]" = torch.ops.aten.reshape.default(sum_350, [1536]);  sum_350 = None
    permute_904: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
    view_1054: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_294, [8, 197, 384]);  mm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    mul_842: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_1054, primals_34);  primals_34 = None
    mul_843: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_842, 384)
    sum_351: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [2], True)
    mul_844: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_842, mul_17);  mul_842 = None
    sum_352: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_844, [2], True);  mul_844 = None
    mul_845: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_17, sum_352);  sum_352 = None
    sub_278: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_843, sum_351);  mul_843 = sum_351 = None
    sub_279: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_278, mul_845);  sub_278 = mul_845 = None
    mul_846: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_80, sub_279);  div_80 = sub_279 = None
    mul_847: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(view_1054, mul_17);  mul_17 = None
    sum_353: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 1]);  mul_847 = None
    sum_354: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_1054, [0, 1]);  view_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_347: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_339, mul_846);  add_339 = mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_1055: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_347, [1576, 384])
    mm_296: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1055, permute_905);  permute_905 = None
    permute_906: "f32[384, 1576]" = torch.ops.aten.permute.default(view_1055, [1, 0])
    mm_297: "f32[384, 384]" = torch.ops.aten.mm.default(permute_906, view_41);  permute_906 = view_41 = None
    permute_907: "f32[384, 384]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    sum_355: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1055, [0], True);  view_1055 = None
    view_1056: "f32[384]" = torch.ops.aten.reshape.default(sum_355, [384]);  sum_355 = None
    permute_908: "f32[384, 384]" = torch.ops.aten.permute.default(permute_907, [1, 0]);  permute_907 = None
    view_1057: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_296, [8, 197, 384]);  mm_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_1058: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_1057, [8, 197, 6, 64]);  view_1057 = None
    permute_909: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_1058, [0, 2, 1, 3]);  view_1058 = None
    clone_295: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(permute_909, memory_format = torch.contiguous_format);  permute_909 = None
    view_1059: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_295, [48, 197, 64]);  clone_295 = None
    bmm_136: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(permute_910, view_1059);  permute_910 = None
    bmm_137: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_1059, permute_911);  view_1059 = permute_911 = None
    view_1060: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_136, [8, 6, 197, 64]);  bmm_136 = None
    view_1061: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_137, [8, 6, 197, 197]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_848: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_1061, alias_46);  view_1061 = None
    sum_356: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_848, [-1], True)
    mul_849: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(alias_46, sum_356);  alias_46 = sum_356 = None
    sub_280: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_850: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(sub_280, 0.125);  sub_280 = None
    view_1062: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(mul_850, [48, 197, 197]);  mul_850 = None
    bmm_138: "f32[48, 64, 197]" = torch.ops.aten.bmm.default(permute_912, view_1062);  permute_912 = None
    bmm_139: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_1062, permute_913);  view_1062 = permute_913 = None
    view_1063: "f32[8, 6, 64, 197]" = torch.ops.aten.reshape.default(bmm_138, [8, 6, 64, 197]);  bmm_138 = None
    view_1064: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_139, [8, 6, 197, 64]);  bmm_139 = None
    permute_914: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_1063, [0, 1, 3, 2]);  view_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_915: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_1060, [0, 2, 1, 3]);  view_1060 = None
    clone_296: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_1065: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_296, [8, 197, 384]);  clone_296 = None
    view_1066: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_1065, [1576, 384]);  view_1065 = None
    permute_916: "f32[384, 1576]" = torch.ops.aten.permute.default(view_1066, [1, 0])
    mm_298: "f32[384, 384]" = torch.ops.aten.mm.default(permute_916, view_28);  permute_916 = None
    permute_917: "f32[384, 384]" = torch.ops.aten.permute.default(mm_298, [1, 0]);  mm_298 = None
    mm_299: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1066, permute_918);  view_1066 = permute_918 = None
    view_1067: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_299, [8, 197, 384]);  mm_299 = None
    permute_919: "f32[384, 384]" = torch.ops.aten.permute.default(permute_917, [1, 0]);  permute_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_35: "f32[16, 6, 197, 64]" = torch.ops.aten.cat.default([view_1064, permute_914]);  view_1064 = permute_914 = None
    view_1068: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.reshape.default(cat_35, [2, 8, 6, 197, 64]);  cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_920: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.permute.default(view_1068, [1, 3, 0, 2, 4]);  view_1068 = None
    clone_297: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.clone.default(permute_920, memory_format = torch.contiguous_format);  permute_920 = None
    view_1069: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_297, [8, 197, 768]);  clone_297 = None
    view_1070: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_1069, [1576, 768]);  view_1069 = None
    permute_921: "f32[768, 1576]" = torch.ops.aten.permute.default(view_1070, [1, 0])
    mm_300: "f32[768, 384]" = torch.ops.aten.mm.default(permute_921, view_28);  permute_921 = view_28 = None
    permute_922: "f32[384, 768]" = torch.ops.aten.permute.default(mm_300, [1, 0]);  mm_300 = None
    mm_301: "f32[1576, 384]" = torch.ops.aten.mm.default(view_1070, permute_923);  view_1070 = permute_923 = None
    view_1071: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_301, [8, 197, 384]);  mm_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_348: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(view_1067, view_1071);  view_1067 = view_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_924: "f32[768, 384]" = torch.ops.aten.permute.default(permute_922, [1, 0]);  permute_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    mul_852: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_348, primals_28);  primals_28 = None
    mul_853: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_852, 384)
    sum_357: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)
    mul_854: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_852, mul_14);  mul_852 = None
    sum_358: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
    mul_855: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_14, sum_358);  sum_358 = None
    sub_282: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(mul_853, sum_357);  mul_853 = sum_357 = None
    sub_283: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(sub_282, mul_855);  sub_282 = mul_855 = None
    div_81: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 384);  rsqrt_5 = None
    mul_856: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(div_81, sub_283);  div_81 = sub_283 = None
    mul_857: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(add_348, mul_14);  mul_14 = None
    sum_359: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
    sum_360: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_348, [0, 1]);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_349: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_347, mul_856);  add_347 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    slice_74: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_349, 1, 0, 1)
    slice_75: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_349, 1, 1, 197);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_298: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_75, memory_format = torch.contiguous_format)
    view_1072: "f32[1568, 384]" = torch.ops.aten.reshape.default(clone_298, [1568, 384]);  clone_298 = None
    mm_302: "f32[1568, 384]" = torch.ops.aten.mm.default(view_1072, permute_925);  permute_925 = None
    permute_926: "f32[384, 1568]" = torch.ops.aten.permute.default(view_1072, [1, 0])
    mm_303: "f32[384, 384]" = torch.ops.aten.mm.default(permute_926, view_26);  permute_926 = view_26 = None
    permute_927: "f32[384, 384]" = torch.ops.aten.permute.default(mm_303, [1, 0]);  mm_303 = None
    sum_361: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1072, [0], True);  view_1072 = None
    view_1073: "f32[384]" = torch.ops.aten.reshape.default(sum_361, [384]);  sum_361 = None
    permute_928: "f32[384, 384]" = torch.ops.aten.permute.default(permute_927, [1, 0]);  permute_927 = None
    view_1074: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_302, [8, 196, 384]);  mm_302 = None
    view_1075: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(view_1074, [1568, 16, 24]);  view_1074 = None
    mul_859: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1075, primals_24);  primals_24 = None
    mul_860: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_859, 24)
    sum_362: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_859, mul_12);  mul_859 = None
    sum_363: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_12, sum_363);  sum_363 = None
    sub_285: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_860, sum_362);  mul_860 = sum_362 = None
    sub_286: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_285, mul_862);  sub_285 = mul_862 = None
    mul_863: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_79, sub_286);  div_79 = sub_286 = None
    mul_864: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1075, mul_12);  mul_12 = None
    sum_364: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
    sum_365: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_1075, [0, 1]);  view_1075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_350: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_344, mul_863);  add_344 = mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_scatter_45: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_75, 1, 1, 9223372036854775807);  slice_75 = None
    slice_scatter_47: "f32[8, 197, 384]" = torch.ops.aten.slice_scatter.default(full_default, slice_74, 1, 0, 1);  full_default = slice_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    add_351: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(slice_scatter_45, slice_scatter_47);  slice_scatter_45 = slice_scatter_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1076: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_350, [25088, 24])
    mm_304: "f32[25088, 96]" = torch.ops.aten.mm.default(view_1076, permute_929);  permute_929 = None
    permute_930: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1076, [1, 0])
    mm_305: "f32[24, 96]" = torch.ops.aten.mm.default(permute_930, view_23);  permute_930 = view_23 = None
    permute_931: "f32[96, 24]" = torch.ops.aten.permute.default(mm_305, [1, 0]);  mm_305 = None
    sum_366: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
    view_1077: "f32[24]" = torch.ops.aten.reshape.default(sum_366, [24]);  sum_366 = None
    permute_932: "f32[24, 96]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_1078: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(mm_304, [1568, 16, 96]);  mm_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_866: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_867: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, view_22)
    mul_868: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_867, -0.5);  mul_867 = None
    exp_47: "f32[1568, 16, 96]" = torch.ops.aten.exp.default(mul_868);  mul_868 = None
    mul_869: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_870: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, mul_869);  view_22 = mul_869 = None
    add_353: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(mul_866, mul_870);  mul_866 = mul_870 = None
    mul_871: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_1078, add_353);  view_1078 = add_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1079: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_871, [25088, 96]);  mul_871 = None
    mm_306: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1079, permute_933);  permute_933 = None
    permute_934: "f32[96, 25088]" = torch.ops.aten.permute.default(view_1079, [1, 0])
    mm_307: "f32[96, 24]" = torch.ops.aten.mm.default(permute_934, view_21);  permute_934 = view_21 = None
    permute_935: "f32[24, 96]" = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
    sum_367: "f32[1, 96]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
    view_1080: "f32[96]" = torch.ops.aten.reshape.default(sum_367, [96]);  sum_367 = None
    permute_936: "f32[96, 24]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_1081: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_306, [1568, 16, 24]);  mm_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    mul_873: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1081, primals_18);  primals_18 = None
    mul_874: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_873, 24)
    sum_368: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
    mul_875: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_873, mul_7);  mul_873 = None
    sum_369: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
    mul_876: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_7, sum_369);  sum_369 = None
    sub_288: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_874, sum_368);  mul_874 = sum_368 = None
    sub_289: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_288, mul_876);  sub_288 = mul_876 = None
    mul_877: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_83, sub_289);  div_83 = sub_289 = None
    mul_878: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(view_1081, mul_7);  mul_7 = None
    sum_370: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
    sum_371: "f32[24]" = torch.ops.aten.sum.dim_IntList(view_1081, [0, 1]);  view_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_354: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_350, mul_877);  add_350 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_1082: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_354, [25088, 24])
    mm_308: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1082, permute_937);  permute_937 = None
    permute_938: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1082, [1, 0])
    mm_309: "f32[24, 24]" = torch.ops.aten.mm.default(permute_938, view_19);  permute_938 = view_19 = None
    permute_939: "f32[24, 24]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    sum_372: "f32[1, 24]" = torch.ops.aten.sum.dim_IntList(view_1082, [0], True);  view_1082 = None
    view_1083: "f32[24]" = torch.ops.aten.reshape.default(sum_372, [24]);  sum_372 = None
    permute_940: "f32[24, 24]" = torch.ops.aten.permute.default(permute_939, [1, 0]);  permute_939 = None
    view_1084: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_308, [1568, 16, 24]);  mm_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    view_1085: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_1084, [1568, 16, 4, 6]);  view_1084 = None
    permute_941: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None
    clone_301: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(permute_941, memory_format = torch.contiguous_format);  permute_941 = None
    view_1086: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_301, [6272, 16, 6]);  clone_301 = None
    bmm_140: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(permute_942, view_1086);  permute_942 = None
    bmm_141: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_1086, permute_943);  view_1086 = permute_943 = None
    view_1087: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_140, [1568, 4, 16, 6]);  bmm_140 = None
    view_1088: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_141, [1568, 4, 16, 16]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    mul_879: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_1088, alias_47);  view_1088 = None
    sum_373: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [-1], True)
    mul_880: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(alias_47, sum_373);  alias_47 = sum_373 = None
    sub_290: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_881: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(sub_290, 0.408248290463863);  sub_290 = None
    view_1089: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(mul_881, [6272, 16, 16]);  mul_881 = None
    bmm_142: "f32[6272, 6, 16]" = torch.ops.aten.bmm.default(permute_944, view_1089);  permute_944 = None
    bmm_143: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_1089, permute_945);  view_1089 = permute_945 = None
    view_1090: "f32[1568, 4, 6, 16]" = torch.ops.aten.reshape.default(bmm_142, [1568, 4, 6, 16]);  bmm_142 = None
    view_1091: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_143, [1568, 4, 16, 6]);  bmm_143 = None
    permute_946: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_1090, [0, 1, 3, 2]);  view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    permute_947: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_1087, [0, 2, 1, 3]);  view_1087 = None
    clone_302: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
    view_1092: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_302, [1568, 16, 24]);  clone_302 = None
    view_1093: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_1092, [25088, 24]);  view_1092 = None
    permute_948: "f32[24, 25088]" = torch.ops.aten.permute.default(view_1093, [1, 0])
    mm_310: "f32[24, 24]" = torch.ops.aten.mm.default(permute_948, view_6);  permute_948 = None
    permute_949: "f32[24, 24]" = torch.ops.aten.permute.default(mm_310, [1, 0]);  mm_310 = None
    mm_311: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1093, permute_950);  view_1093 = permute_950 = None
    view_1094: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_311, [1568, 16, 24]);  mm_311 = None
    permute_951: "f32[24, 24]" = torch.ops.aten.permute.default(permute_949, [1, 0]);  permute_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    cat_36: "f32[3136, 4, 16, 6]" = torch.ops.aten.cat.default([view_1091, permute_946]);  view_1091 = permute_946 = None
    view_1095: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.reshape.default(cat_36, [2, 1568, 4, 16, 6]);  cat_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_952: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.permute.default(view_1095, [1, 3, 0, 2, 4]);  view_1095 = None
    clone_303: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
    view_1096: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(clone_303, [1568, 16, 48]);  clone_303 = None
    view_1097: "f32[25088, 48]" = torch.ops.aten.reshape.default(view_1096, [25088, 48]);  view_1096 = None
    permute_953: "f32[48, 25088]" = torch.ops.aten.permute.default(view_1097, [1, 0])
    mm_312: "f32[48, 24]" = torch.ops.aten.mm.default(permute_953, view_6);  permute_953 = view_6 = None
    permute_954: "f32[24, 48]" = torch.ops.aten.permute.default(mm_312, [1, 0]);  mm_312 = None
    mm_313: "f32[25088, 24]" = torch.ops.aten.mm.default(view_1097, permute_955);  view_1097 = permute_955 = None
    view_1098: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_313, [1568, 16, 24]);  mm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    add_355: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(view_1094, view_1098);  view_1094 = view_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_956: "f32[48, 24]" = torch.ops.aten.permute.default(permute_954, [1, 0]);  permute_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    mul_883: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_355, primals_12);  primals_12 = None
    mul_884: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_883, 24)
    sum_374: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_883, [2], True)
    mul_885: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_883, mul_4);  mul_883 = None
    sum_375: "f32[1568, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_885, [2], True);  mul_885 = None
    mul_886: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_4, sum_375);  sum_375 = None
    sub_292: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(mul_884, sum_374);  mul_884 = sum_374 = None
    sub_293: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(sub_292, mul_886);  sub_292 = mul_886 = None
    div_84: "f32[1568, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 24);  rsqrt_2 = None
    mul_887: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(div_84, sub_293);  div_84 = sub_293 = None
    mul_888: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(add_355, mul_4);  mul_4 = None
    sum_376: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 1]);  mul_888 = None
    sum_377: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_355, [0, 1]);  add_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_356: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_354, mul_887);  add_354 = mul_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:313, code: patch_embed = patch_embed + self.patch_pos
    sum_378: "f32[1, 197, 384]" = torch.ops.aten.sum.dim_IntList(add_351, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:312, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    slice_76: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_351, 1, 0, 1)
    slice_77: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_351, 1, 1, 197);  add_351 = None
    sum_379: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(slice_76, [0], True);  slice_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    clone_305: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_77, memory_format = torch.contiguous_format);  slice_77 = None
    mul_890: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_305, primals_10);  primals_10 = None
    mul_891: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_890, 384)
    sum_380: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_890, [2], True)
    mul_892: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_890, mul_2);  mul_890 = None
    sum_381: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [2], True);  mul_892 = None
    mul_893: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_2, sum_381);  sum_381 = None
    sub_295: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_891, sum_380);  mul_891 = sum_380 = None
    sub_296: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_295, mul_893);  sub_295 = mul_893 = None
    div_85: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 384);  rsqrt_1 = None
    mul_894: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_85, sub_296);  div_85 = sub_296 = None
    mul_895: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(clone_305, mul_2);  mul_2 = None
    sum_382: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 1]);  mul_895 = None
    sum_383: "f32[384]" = torch.ops.aten.sum.dim_IntList(clone_305, [0, 1]);  clone_305 = None
    view_1099: "f32[1568, 384]" = torch.ops.aten.reshape.default(mul_894, [1568, 384]);  mul_894 = None
    mm_314: "f32[1568, 384]" = torch.ops.aten.mm.default(view_1099, permute_957);  permute_957 = None
    permute_958: "f32[384, 1568]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_315: "f32[384, 384]" = torch.ops.aten.mm.default(permute_958, view_4);  permute_958 = view_4 = None
    permute_959: "f32[384, 384]" = torch.ops.aten.permute.default(mm_315, [1, 0]);  mm_315 = None
    sum_384: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
    view_1100: "f32[384]" = torch.ops.aten.reshape.default(sum_384, [384]);  sum_384 = None
    permute_960: "f32[384, 384]" = torch.ops.aten.permute.default(permute_959, [1, 0]);  permute_959 = None
    view_1101: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(mm_314, [8, 196, 384]);  mm_314 = None
    mul_897: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_1101, primals_6);  primals_6 = None
    mul_898: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_897, 384)
    sum_385: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_897, [2], True)
    mul_899: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_897, mul);  mul_897 = None
    sum_386: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_899, [2], True);  mul_899 = None
    mul_900: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, sum_386);  sum_386 = None
    sub_298: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(mul_898, sum_385);  mul_898 = sum_385 = None
    sub_299: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(sub_298, mul_900);  sub_298 = mul_900 = None
    div_86: "f32[8, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt, 384);  rsqrt = None
    mul_901: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(div_86, sub_299);  div_86 = sub_299 = None
    mul_902: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(view_1101, mul);  mul = None
    sum_387: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_902, [0, 1]);  mul_902 = None
    sum_388: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_1101, [0, 1]);  view_1101 = None
    view_1102: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mul_901, [1568, 16, 24]);  mul_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    add_357: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_356, view_1102);  add_356 = view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:185, code: x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
    permute_961: "f32[1568, 24, 16]" = torch.ops.aten.permute.default(add_357, [0, 2, 1]);  add_357 = None
    view_1103: "f32[1568, 24, 4, 4]" = torch.ops.aten.reshape.default(permute_961, [1568, 24, 4, 4]);  permute_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:184, code: x = x + pixel_pos
    sum_389: "f32[1, 24, 4, 4]" = torch.ops.aten.sum.dim_IntList(view_1103, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:183, code: x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
    clone_306: "f32[1568, 24, 4, 4]" = torch.ops.aten.clone.default(view_1103, memory_format = torch.contiguous_format);  view_1103 = None
    view_1104: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(clone_306, [8, 196, 384]);  clone_306 = None
    permute_962: "f32[8, 384, 196]" = torch.ops.aten.permute.default(view_1104, [0, 2, 1]);  view_1104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:182, code: x = self.unfold(x)
    view_1105: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.reshape.default(permute_962, [8, 24, 4, 4, 14, 14]);  permute_962 = None
    permute_963: "f32[8, 24, 4, 14, 4, 14]" = torch.ops.aten.permute.default(view_1105, [0, 1, 2, 4, 3, 5]);  view_1105 = None
    full_default_50: "f32[8, 24, 56, 56]" = torch.ops.aten.full.default([8, 24, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[8, 24, 56, 56]" = torch.ops.prims._unsafe_index_put_.default(full_default_50, [None, None, unsqueeze_5, add], permute_963, True);  full_default_50 = unsqueeze_5 = add = permute_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:181, code: x = self.proj(x)
    sum_390: "f32[24]" = torch.ops.aten.sum.dim_IntList(_unsafe_index_put, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(_unsafe_index_put, primals_352, primals_4, [24], [4, 4], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  _unsafe_index_put = primals_352 = primals_4 = None
    getitem_175: "f32[24, 3, 7, 7]" = convolution_backward[1];  convolution_backward = None
    return [sum_389, sum_379, sum_378, getitem_175, sum_390, sum_387, sum_388, permute_960, view_1100, sum_382, sum_383, sum_376, sum_377, permute_956, permute_951, permute_940, view_1083, sum_370, sum_371, permute_936, view_1080, permute_932, view_1077, sum_364, sum_365, permute_928, view_1073, sum_359, sum_360, permute_924, permute_919, permute_908, view_1056, sum_353, sum_354, permute_904, view_1053, permute_900, view_1050, sum_347, sum_348, permute_896, permute_891, permute_880, view_1033, sum_341, sum_342, permute_876, view_1030, permute_872, view_1027, sum_335, sum_336, permute_868, view_1023, sum_330, sum_331, permute_864, permute_859, permute_848, view_1006, sum_324, sum_325, permute_844, view_1003, permute_840, view_1000, sum_318, sum_319, permute_836, permute_831, permute_820, view_983, sum_312, sum_313, permute_816, view_980, permute_812, view_977, sum_306, sum_307, permute_808, view_973, sum_301, sum_302, permute_804, permute_799, permute_788, view_956, sum_295, sum_296, permute_784, view_953, permute_780, view_950, sum_289, sum_290, permute_776, permute_771, permute_760, view_933, sum_283, sum_284, permute_756, view_930, permute_752, view_927, sum_277, sum_278, permute_748, view_923, sum_272, sum_273, permute_744, permute_739, permute_728, view_906, sum_266, sum_267, permute_724, view_903, permute_720, view_900, sum_260, sum_261, permute_716, permute_711, permute_700, view_883, sum_254, sum_255, permute_696, view_880, permute_692, view_877, sum_248, sum_249, permute_688, view_873, sum_243, sum_244, permute_684, permute_679, permute_668, view_856, sum_237, sum_238, permute_664, view_853, permute_660, view_850, sum_231, sum_232, permute_656, permute_651, permute_640, view_833, sum_225, sum_226, permute_636, view_830, permute_632, view_827, sum_219, sum_220, permute_628, view_823, sum_214, sum_215, permute_624, permute_619, permute_608, view_806, sum_208, sum_209, permute_604, view_803, permute_600, view_800, sum_202, sum_203, permute_596, permute_591, permute_580, view_783, sum_196, sum_197, permute_576, view_780, permute_572, view_777, sum_190, sum_191, permute_568, view_773, sum_185, sum_186, permute_564, permute_559, permute_548, view_756, sum_179, sum_180, permute_544, view_753, permute_540, view_750, sum_173, sum_174, permute_536, permute_531, permute_520, view_733, sum_167, sum_168, permute_516, view_730, permute_512, view_727, sum_161, sum_162, permute_508, view_723, sum_156, sum_157, permute_504, permute_499, permute_488, view_706, sum_150, sum_151, permute_484, view_703, permute_480, view_700, sum_144, sum_145, permute_476, permute_471, permute_460, view_683, sum_138, sum_139, permute_456, view_680, permute_452, view_677, sum_132, sum_133, permute_448, view_673, sum_127, sum_128, permute_444, permute_439, permute_428, view_656, sum_121, sum_122, permute_424, view_653, permute_420, view_650, sum_115, sum_116, permute_416, permute_411, permute_400, view_633, sum_109, sum_110, permute_396, view_630, permute_392, view_627, sum_103, sum_104, permute_388, view_623, sum_98, sum_99, permute_384, permute_379, permute_368, view_606, sum_92, sum_93, permute_364, view_603, permute_360, view_600, sum_86, sum_87, permute_356, permute_351, permute_340, view_583, sum_80, sum_81, permute_336, view_580, permute_332, view_577, sum_74, sum_75, permute_328, view_573, sum_69, sum_70, permute_324, permute_319, permute_308, view_556, sum_63, sum_64, permute_304, view_553, permute_300, view_550, sum_57, sum_58, permute_296, permute_291, permute_280, view_533, sum_51, sum_52, permute_276, view_530, permute_272, view_527, sum_45, sum_46, permute_268, view_523, sum_40, sum_41, permute_264, permute_259, permute_248, view_506, sum_34, sum_35, permute_244, view_503, permute_240, view_500, sum_28, sum_29, permute_236, view_498, None]
    