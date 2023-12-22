from __future__ import annotations



def forward(self, primals_4: "f32[1024]", primals_14: "f32[1024]", primals_20: "f32[1024]", primals_30: "f32[1024]", primals_36: "f32[1024]", primals_46: "f32[1024]", primals_52: "f32[1024]", primals_62: "f32[1024]", primals_68: "f32[1024]", primals_78: "f32[1024]", primals_84: "f32[1024]", primals_94: "f32[1024]", primals_100: "f32[1024]", primals_110: "f32[1024]", primals_116: "f32[1024]", primals_126: "f32[1024]", primals_132: "f32[1024]", primals_142: "f32[1024]", primals_148: "f32[1024]", primals_158: "f32[1024]", primals_164: "f32[1024]", primals_174: "f32[1024]", primals_180: "f32[1024]", primals_190: "f32[1024]", primals_196: "f32[1024]", primals_206: "f32[1024]", primals_212: "f32[1024]", primals_222: "f32[1024]", primals_228: "f32[1024]", primals_238: "f32[1024]", primals_244: "f32[1024]", primals_254: "f32[1024]", primals_260: "f32[1024]", primals_270: "f32[1024]", primals_276: "f32[1024]", primals_286: "f32[1024]", primals_292: "f32[1024]", primals_302: "f32[1024]", primals_308: "f32[1024]", primals_318: "f32[1024]", primals_324: "f32[1024]", primals_334: "f32[1024]", primals_340: "f32[1024]", primals_350: "f32[1024]", primals_356: "f32[1024]", primals_366: "f32[1024]", primals_372: "f32[1024]", primals_382: "f32[1024]", primals_388: "f32[1024]", primals_392: "f32[1024]", primals_398: "i64[1, 512]", full_default: "i64[1, 512]", slice_3: "i64[1, 512]", getitem_1: "b8[1, 512, 1024]", mul_1: "f32[1, 512, 1024]", view: "f32[512, 1024]", getitem_293: "b8[1, 16, 512, 512]", permute_default_139: "f32[16, 512, 512]", permute_default_140: "f32[16, 64, 512]", alias_default_47: "f32[1, 16, 512, 512]", permute_default_141: "f32[16, 64, 512]", permute_default_142: "f32[16, 512, 64]", view_16: "f32[512, 1024]", getitem_7: "b8[1, 512, 1024]", mul_3: "f32[1, 512, 1024]", view_18: "f32[512, 1024]", addmm_4: "f32[512, 4096]", view_20: "f32[512, 4096]", getitem_11: "b8[1, 512, 1024]", mul_8: "f32[1, 512, 1024]", view_22: "f32[512, 1024]", getitem_291: "b8[1, 16, 512, 512]", permute_default_133: "f32[16, 512, 512]", permute_default_134: "f32[16, 64, 512]", alias_default_45: "f32[1, 16, 512, 512]", permute_default_135: "f32[16, 64, 512]", permute_default_136: "f32[16, 512, 64]", view_38: "f32[512, 1024]", getitem_17: "b8[1, 512, 1024]", mul_10: "f32[1, 512, 1024]", view_40: "f32[512, 1024]", addmm_10: "f32[512, 4096]", view_42: "f32[512, 4096]", getitem_21: "b8[1, 512, 1024]", mul_15: "f32[1, 512, 1024]", view_44: "f32[512, 1024]", getitem_289: "b8[1, 16, 512, 512]", permute_default_127: "f32[16, 512, 512]", permute_default_128: "f32[16, 64, 512]", alias_default_43: "f32[1, 16, 512, 512]", permute_default_129: "f32[16, 64, 512]", permute_default_130: "f32[16, 512, 64]", view_60: "f32[512, 1024]", getitem_27: "b8[1, 512, 1024]", mul_17: "f32[1, 512, 1024]", view_62: "f32[512, 1024]", addmm_16: "f32[512, 4096]", view_64: "f32[512, 4096]", getitem_31: "b8[1, 512, 1024]", mul_22: "f32[1, 512, 1024]", view_66: "f32[512, 1024]", getitem_287: "b8[1, 16, 512, 512]", permute_default_121: "f32[16, 512, 512]", permute_default_122: "f32[16, 64, 512]", alias_default_41: "f32[1, 16, 512, 512]", permute_default_123: "f32[16, 64, 512]", permute_default_124: "f32[16, 512, 64]", view_82: "f32[512, 1024]", getitem_37: "b8[1, 512, 1024]", mul_24: "f32[1, 512, 1024]", view_84: "f32[512, 1024]", addmm_22: "f32[512, 4096]", view_86: "f32[512, 4096]", getitem_41: "b8[1, 512, 1024]", mul_29: "f32[1, 512, 1024]", view_88: "f32[512, 1024]", getitem_285: "b8[1, 16, 512, 512]", permute_default_115: "f32[16, 512, 512]", permute_default_116: "f32[16, 64, 512]", alias_default_39: "f32[1, 16, 512, 512]", permute_default_117: "f32[16, 64, 512]", permute_default_118: "f32[16, 512, 64]", view_104: "f32[512, 1024]", getitem_47: "b8[1, 512, 1024]", mul_31: "f32[1, 512, 1024]", view_106: "f32[512, 1024]", addmm_28: "f32[512, 4096]", view_108: "f32[512, 4096]", getitem_51: "b8[1, 512, 1024]", mul_36: "f32[1, 512, 1024]", view_110: "f32[512, 1024]", getitem_283: "b8[1, 16, 512, 512]", permute_default_109: "f32[16, 512, 512]", permute_default_110: "f32[16, 64, 512]", alias_default_37: "f32[1, 16, 512, 512]", permute_default_111: "f32[16, 64, 512]", permute_default_112: "f32[16, 512, 64]", view_126: "f32[512, 1024]", getitem_57: "b8[1, 512, 1024]", mul_38: "f32[1, 512, 1024]", view_128: "f32[512, 1024]", addmm_34: "f32[512, 4096]", view_130: "f32[512, 4096]", getitem_61: "b8[1, 512, 1024]", mul_43: "f32[1, 512, 1024]", view_132: "f32[512, 1024]", getitem_281: "b8[1, 16, 512, 512]", permute_default_103: "f32[16, 512, 512]", permute_default_104: "f32[16, 64, 512]", alias_default_35: "f32[1, 16, 512, 512]", permute_default_105: "f32[16, 64, 512]", permute_default_106: "f32[16, 512, 64]", view_148: "f32[512, 1024]", getitem_67: "b8[1, 512, 1024]", mul_45: "f32[1, 512, 1024]", view_150: "f32[512, 1024]", addmm_40: "f32[512, 4096]", view_152: "f32[512, 4096]", getitem_71: "b8[1, 512, 1024]", mul_50: "f32[1, 512, 1024]", view_154: "f32[512, 1024]", getitem_279: "b8[1, 16, 512, 512]", permute_default_97: "f32[16, 512, 512]", permute_default_98: "f32[16, 64, 512]", alias_default_33: "f32[1, 16, 512, 512]", permute_default_99: "f32[16, 64, 512]", permute_default_100: "f32[16, 512, 64]", view_170: "f32[512, 1024]", getitem_77: "b8[1, 512, 1024]", mul_52: "f32[1, 512, 1024]", view_172: "f32[512, 1024]", addmm_46: "f32[512, 4096]", view_174: "f32[512, 4096]", getitem_81: "b8[1, 512, 1024]", mul_57: "f32[1, 512, 1024]", view_176: "f32[512, 1024]", getitem_277: "b8[1, 16, 512, 512]", permute_default_91: "f32[16, 512, 512]", permute_default_92: "f32[16, 64, 512]", alias_default_31: "f32[1, 16, 512, 512]", permute_default_93: "f32[16, 64, 512]", permute_default_94: "f32[16, 512, 64]", view_192: "f32[512, 1024]", getitem_87: "b8[1, 512, 1024]", mul_59: "f32[1, 512, 1024]", view_194: "f32[512, 1024]", addmm_52: "f32[512, 4096]", view_196: "f32[512, 4096]", getitem_91: "b8[1, 512, 1024]", mul_64: "f32[1, 512, 1024]", view_198: "f32[512, 1024]", getitem_275: "b8[1, 16, 512, 512]", permute_default_85: "f32[16, 512, 512]", permute_default_86: "f32[16, 64, 512]", alias_default_29: "f32[1, 16, 512, 512]", permute_default_87: "f32[16, 64, 512]", permute_default_88: "f32[16, 512, 64]", view_214: "f32[512, 1024]", getitem_97: "b8[1, 512, 1024]", mul_66: "f32[1, 512, 1024]", view_216: "f32[512, 1024]", addmm_58: "f32[512, 4096]", view_218: "f32[512, 4096]", getitem_101: "b8[1, 512, 1024]", mul_71: "f32[1, 512, 1024]", view_220: "f32[512, 1024]", getitem_273: "b8[1, 16, 512, 512]", permute_default_79: "f32[16, 512, 512]", permute_default_80: "f32[16, 64, 512]", alias_default_27: "f32[1, 16, 512, 512]", permute_default_81: "f32[16, 64, 512]", permute_default_82: "f32[16, 512, 64]", view_236: "f32[512, 1024]", getitem_107: "b8[1, 512, 1024]", mul_73: "f32[1, 512, 1024]", view_238: "f32[512, 1024]", addmm_64: "f32[512, 4096]", view_240: "f32[512, 4096]", getitem_111: "b8[1, 512, 1024]", mul_78: "f32[1, 512, 1024]", view_242: "f32[512, 1024]", getitem_271: "b8[1, 16, 512, 512]", permute_default_73: "f32[16, 512, 512]", permute_default_74: "f32[16, 64, 512]", alias_default_25: "f32[1, 16, 512, 512]", permute_default_75: "f32[16, 64, 512]", permute_default_76: "f32[16, 512, 64]", view_258: "f32[512, 1024]", getitem_117: "b8[1, 512, 1024]", mul_80: "f32[1, 512, 1024]", view_260: "f32[512, 1024]", addmm_70: "f32[512, 4096]", view_262: "f32[512, 4096]", getitem_121: "b8[1, 512, 1024]", mul_85: "f32[1, 512, 1024]", view_264: "f32[512, 1024]", getitem_269: "b8[1, 16, 512, 512]", permute_default_67: "f32[16, 512, 512]", permute_default_68: "f32[16, 64, 512]", alias_default_23: "f32[1, 16, 512, 512]", permute_default_69: "f32[16, 64, 512]", permute_default_70: "f32[16, 512, 64]", view_280: "f32[512, 1024]", getitem_127: "b8[1, 512, 1024]", mul_87: "f32[1, 512, 1024]", view_282: "f32[512, 1024]", addmm_76: "f32[512, 4096]", view_284: "f32[512, 4096]", getitem_131: "b8[1, 512, 1024]", mul_92: "f32[1, 512, 1024]", view_286: "f32[512, 1024]", getitem_267: "b8[1, 16, 512, 512]", permute_default_61: "f32[16, 512, 512]", permute_default_62: "f32[16, 64, 512]", alias_default_21: "f32[1, 16, 512, 512]", permute_default_63: "f32[16, 64, 512]", permute_default_64: "f32[16, 512, 64]", view_302: "f32[512, 1024]", getitem_137: "b8[1, 512, 1024]", mul_94: "f32[1, 512, 1024]", view_304: "f32[512, 1024]", addmm_82: "f32[512, 4096]", view_306: "f32[512, 4096]", getitem_141: "b8[1, 512, 1024]", mul_99: "f32[1, 512, 1024]", view_308: "f32[512, 1024]", getitem_265: "b8[1, 16, 512, 512]", permute_default_55: "f32[16, 512, 512]", permute_default_56: "f32[16, 64, 512]", alias_default_19: "f32[1, 16, 512, 512]", permute_default_57: "f32[16, 64, 512]", permute_default_58: "f32[16, 512, 64]", view_324: "f32[512, 1024]", getitem_147: "b8[1, 512, 1024]", mul_101: "f32[1, 512, 1024]", view_326: "f32[512, 1024]", addmm_88: "f32[512, 4096]", view_328: "f32[512, 4096]", getitem_151: "b8[1, 512, 1024]", mul_106: "f32[1, 512, 1024]", view_330: "f32[512, 1024]", getitem_263: "b8[1, 16, 512, 512]", permute_default_49: "f32[16, 512, 512]", permute_default_50: "f32[16, 64, 512]", alias_default_17: "f32[1, 16, 512, 512]", permute_default_51: "f32[16, 64, 512]", permute_default_52: "f32[16, 512, 64]", view_346: "f32[512, 1024]", getitem_157: "b8[1, 512, 1024]", mul_108: "f32[1, 512, 1024]", view_348: "f32[512, 1024]", addmm_94: "f32[512, 4096]", view_350: "f32[512, 4096]", getitem_161: "b8[1, 512, 1024]", mul_113: "f32[1, 512, 1024]", view_352: "f32[512, 1024]", getitem_261: "b8[1, 16, 512, 512]", permute_default_43: "f32[16, 512, 512]", permute_default_44: "f32[16, 64, 512]", alias_default_15: "f32[1, 16, 512, 512]", permute_default_45: "f32[16, 64, 512]", permute_default_46: "f32[16, 512, 64]", view_368: "f32[512, 1024]", getitem_167: "b8[1, 512, 1024]", mul_115: "f32[1, 512, 1024]", view_370: "f32[512, 1024]", addmm_100: "f32[512, 4096]", view_372: "f32[512, 4096]", getitem_171: "b8[1, 512, 1024]", mul_120: "f32[1, 512, 1024]", view_374: "f32[512, 1024]", getitem_259: "b8[1, 16, 512, 512]", permute_default_37: "f32[16, 512, 512]", permute_default_38: "f32[16, 64, 512]", alias_default_13: "f32[1, 16, 512, 512]", permute_default_39: "f32[16, 64, 512]", permute_default_40: "f32[16, 512, 64]", view_390: "f32[512, 1024]", getitem_177: "b8[1, 512, 1024]", mul_122: "f32[1, 512, 1024]", view_392: "f32[512, 1024]", addmm_106: "f32[512, 4096]", view_394: "f32[512, 4096]", getitem_181: "b8[1, 512, 1024]", mul_127: "f32[1, 512, 1024]", view_396: "f32[512, 1024]", getitem_257: "b8[1, 16, 512, 512]", permute_default_31: "f32[16, 512, 512]", permute_default_32: "f32[16, 64, 512]", alias_default_11: "f32[1, 16, 512, 512]", permute_default_33: "f32[16, 64, 512]", permute_default_34: "f32[16, 512, 64]", view_412: "f32[512, 1024]", getitem_187: "b8[1, 512, 1024]", mul_129: "f32[1, 512, 1024]", view_414: "f32[512, 1024]", addmm_112: "f32[512, 4096]", view_416: "f32[512, 4096]", getitem_191: "b8[1, 512, 1024]", mul_134: "f32[1, 512, 1024]", view_418: "f32[512, 1024]", getitem_255: "b8[1, 16, 512, 512]", permute_default_25: "f32[16, 512, 512]", permute_default_26: "f32[16, 64, 512]", alias_default_9: "f32[1, 16, 512, 512]", permute_default_27: "f32[16, 64, 512]", permute_default_28: "f32[16, 512, 64]", view_434: "f32[512, 1024]", getitem_197: "b8[1, 512, 1024]", mul_136: "f32[1, 512, 1024]", view_436: "f32[512, 1024]", addmm_118: "f32[512, 4096]", view_438: "f32[512, 4096]", getitem_201: "b8[1, 512, 1024]", mul_141: "f32[1, 512, 1024]", view_440: "f32[512, 1024]", getitem_253: "b8[1, 16, 512, 512]", permute_default_19: "f32[16, 512, 512]", permute_default_20: "f32[16, 64, 512]", alias_default_7: "f32[1, 16, 512, 512]", permute_default_21: "f32[16, 64, 512]", permute_default_22: "f32[16, 512, 64]", view_456: "f32[512, 1024]", getitem_207: "b8[1, 512, 1024]", mul_143: "f32[1, 512, 1024]", view_458: "f32[512, 1024]", addmm_124: "f32[512, 4096]", view_460: "f32[512, 4096]", getitem_211: "b8[1, 512, 1024]", mul_148: "f32[1, 512, 1024]", view_462: "f32[512, 1024]", getitem_251: "b8[1, 16, 512, 512]", permute_default_13: "f32[16, 512, 512]", permute_default_14: "f32[16, 64, 512]", alias_default_5: "f32[1, 16, 512, 512]", permute_default_15: "f32[16, 64, 512]", permute_default_16: "f32[16, 512, 64]", view_478: "f32[512, 1024]", getitem_217: "b8[1, 512, 1024]", mul_150: "f32[1, 512, 1024]", view_480: "f32[512, 1024]", addmm_130: "f32[512, 4096]", view_482: "f32[512, 4096]", getitem_221: "b8[1, 512, 1024]", mul_155: "f32[1, 512, 1024]", view_484: "f32[512, 1024]", getitem_249: "b8[1, 16, 512, 512]", permute_default_7: "f32[16, 512, 512]", permute_default_8: "f32[16, 64, 512]", alias_default_3: "f32[1, 16, 512, 512]", permute_default_9: "f32[16, 64, 512]", permute_default_10: "f32[16, 512, 64]", view_500: "f32[512, 1024]", getitem_227: "b8[1, 512, 1024]", mul_157: "f32[1, 512, 1024]", view_502: "f32[512, 1024]", addmm_136: "f32[512, 4096]", view_504: "f32[512, 4096]", getitem_231: "b8[1, 512, 1024]", mul_162: "f32[1, 512, 1024]", view_506: "f32[512, 1024]", getitem_247: "b8[1, 16, 512, 512]", permute_default_1: "f32[16, 512, 512]", permute_default_2: "f32[16, 64, 512]", alias_default_1: "f32[1, 16, 512, 512]", permute_default_3: "f32[16, 64, 512]", permute_default_4: "f32[16, 512, 64]", view_522: "f32[512, 1024]", getitem_237: "b8[1, 512, 1024]", mul_164: "f32[1, 512, 1024]", view_524: "f32[512, 1024]", addmm_142: "f32[512, 4096]", view_526: "f32[512, 4096]", getitem_241: "b8[1, 512, 1024]", mul_169: "f32[1, 512, 1024]", view_528: "f32[512, 1024]", addmm_144: "f32[512, 1024]", mul_174: "f32[1, 512, 1024]", view_530: "f32[512, 1024]", sub_76: "f32[511, 29056]", convert_element_type: "f32[]", ne_3: "b8[511, 1]", where_2: "i64[511, 1]", permute_266: "f32[29056, 1024]", div_50: "f32[1, 512, 1]", permute_270: "f32[1024, 1024]", div_51: "f32[1, 512, 1]", permute_274: "f32[1024, 4096]", permute_278: "f32[4096, 1024]", div_52: "f32[1, 512, 1]", permute_282: "f32[1024, 1024]", permute_294: "f32[1024, 1024]", permute_299: "f32[1024, 1024]", permute_303: "f32[1024, 1024]", div_54: "f32[1, 512, 1]", permute_307: "f32[1024, 4096]", permute_311: "f32[4096, 1024]", div_55: "f32[1, 512, 1]", permute_315: "f32[1024, 1024]", permute_327: "f32[1024, 1024]", permute_332: "f32[1024, 1024]", permute_336: "f32[1024, 1024]", div_57: "f32[1, 512, 1]", permute_340: "f32[1024, 4096]", permute_344: "f32[4096, 1024]", div_58: "f32[1, 512, 1]", permute_348: "f32[1024, 1024]", permute_360: "f32[1024, 1024]", permute_365: "f32[1024, 1024]", permute_369: "f32[1024, 1024]", div_60: "f32[1, 512, 1]", permute_373: "f32[1024, 4096]", permute_377: "f32[4096, 1024]", div_61: "f32[1, 512, 1]", permute_381: "f32[1024, 1024]", permute_393: "f32[1024, 1024]", permute_398: "f32[1024, 1024]", permute_402: "f32[1024, 1024]", div_63: "f32[1, 512, 1]", permute_406: "f32[1024, 4096]", permute_410: "f32[4096, 1024]", div_64: "f32[1, 512, 1]", permute_414: "f32[1024, 1024]", permute_426: "f32[1024, 1024]", permute_431: "f32[1024, 1024]", permute_435: "f32[1024, 1024]", div_66: "f32[1, 512, 1]", permute_439: "f32[1024, 4096]", permute_443: "f32[4096, 1024]", div_67: "f32[1, 512, 1]", permute_447: "f32[1024, 1024]", permute_459: "f32[1024, 1024]", permute_464: "f32[1024, 1024]", permute_468: "f32[1024, 1024]", div_69: "f32[1, 512, 1]", permute_472: "f32[1024, 4096]", permute_476: "f32[4096, 1024]", div_70: "f32[1, 512, 1]", permute_480: "f32[1024, 1024]", permute_492: "f32[1024, 1024]", permute_497: "f32[1024, 1024]", permute_501: "f32[1024, 1024]", div_72: "f32[1, 512, 1]", permute_505: "f32[1024, 4096]", permute_509: "f32[4096, 1024]", div_73: "f32[1, 512, 1]", permute_513: "f32[1024, 1024]", permute_525: "f32[1024, 1024]", permute_530: "f32[1024, 1024]", permute_534: "f32[1024, 1024]", div_75: "f32[1, 512, 1]", permute_538: "f32[1024, 4096]", permute_542: "f32[4096, 1024]", div_76: "f32[1, 512, 1]", permute_546: "f32[1024, 1024]", permute_558: "f32[1024, 1024]", permute_563: "f32[1024, 1024]", permute_567: "f32[1024, 1024]", div_78: "f32[1, 512, 1]", permute_571: "f32[1024, 4096]", permute_575: "f32[4096, 1024]", div_79: "f32[1, 512, 1]", permute_579: "f32[1024, 1024]", permute_591: "f32[1024, 1024]", permute_596: "f32[1024, 1024]", permute_600: "f32[1024, 1024]", div_81: "f32[1, 512, 1]", permute_604: "f32[1024, 4096]", permute_608: "f32[4096, 1024]", div_82: "f32[1, 512, 1]", permute_612: "f32[1024, 1024]", permute_624: "f32[1024, 1024]", permute_629: "f32[1024, 1024]", permute_633: "f32[1024, 1024]", div_84: "f32[1, 512, 1]", permute_637: "f32[1024, 4096]", permute_641: "f32[4096, 1024]", div_85: "f32[1, 512, 1]", permute_645: "f32[1024, 1024]", permute_657: "f32[1024, 1024]", permute_662: "f32[1024, 1024]", permute_666: "f32[1024, 1024]", div_87: "f32[1, 512, 1]", permute_670: "f32[1024, 4096]", permute_674: "f32[4096, 1024]", div_88: "f32[1, 512, 1]", permute_678: "f32[1024, 1024]", permute_690: "f32[1024, 1024]", permute_695: "f32[1024, 1024]", permute_699: "f32[1024, 1024]", div_90: "f32[1, 512, 1]", permute_703: "f32[1024, 4096]", permute_707: "f32[4096, 1024]", div_91: "f32[1, 512, 1]", permute_711: "f32[1024, 1024]", permute_723: "f32[1024, 1024]", permute_728: "f32[1024, 1024]", permute_732: "f32[1024, 1024]", div_93: "f32[1, 512, 1]", permute_736: "f32[1024, 4096]", permute_740: "f32[4096, 1024]", div_94: "f32[1, 512, 1]", permute_744: "f32[1024, 1024]", permute_756: "f32[1024, 1024]", permute_761: "f32[1024, 1024]", permute_765: "f32[1024, 1024]", div_96: "f32[1, 512, 1]", permute_769: "f32[1024, 4096]", permute_773: "f32[4096, 1024]", div_97: "f32[1, 512, 1]", permute_777: "f32[1024, 1024]", permute_789: "f32[1024, 1024]", permute_794: "f32[1024, 1024]", permute_798: "f32[1024, 1024]", div_99: "f32[1, 512, 1]", permute_802: "f32[1024, 4096]", permute_806: "f32[4096, 1024]", div_100: "f32[1, 512, 1]", permute_810: "f32[1024, 1024]", permute_822: "f32[1024, 1024]", permute_827: "f32[1024, 1024]", permute_831: "f32[1024, 1024]", div_102: "f32[1, 512, 1]", permute_835: "f32[1024, 4096]", permute_839: "f32[4096, 1024]", div_103: "f32[1, 512, 1]", permute_843: "f32[1024, 1024]", permute_855: "f32[1024, 1024]", permute_860: "f32[1024, 1024]", permute_864: "f32[1024, 1024]", div_105: "f32[1, 512, 1]", permute_868: "f32[1024, 4096]", permute_872: "f32[4096, 1024]", div_106: "f32[1, 512, 1]", permute_876: "f32[1024, 1024]", permute_888: "f32[1024, 1024]", permute_893: "f32[1024, 1024]", permute_897: "f32[1024, 1024]", div_108: "f32[1, 512, 1]", permute_901: "f32[1024, 4096]", permute_905: "f32[4096, 1024]", div_109: "f32[1, 512, 1]", permute_909: "f32[1024, 1024]", permute_921: "f32[1024, 1024]", permute_926: "f32[1024, 1024]", permute_930: "f32[1024, 1024]", div_111: "f32[1, 512, 1]", permute_934: "f32[1024, 4096]", permute_938: "f32[4096, 1024]", div_112: "f32[1, 512, 1]", permute_942: "f32[1024, 1024]", permute_954: "f32[1024, 1024]", permute_959: "f32[1024, 1024]", permute_963: "f32[1024, 1024]", div_114: "f32[1, 512, 1]", permute_967: "f32[1024, 4096]", permute_971: "f32[4096, 1024]", div_115: "f32[1, 512, 1]", permute_975: "f32[1024, 1024]", permute_987: "f32[1024, 1024]", permute_992: "f32[1024, 1024]", permute_996: "f32[1024, 1024]", div_117: "f32[1, 512, 1]", permute_1000: "f32[1024, 4096]", permute_1004: "f32[4096, 1024]", div_118: "f32[1, 512, 1]", permute_1008: "f32[1024, 1024]", permute_1020: "f32[1024, 1024]", permute_1025: "f32[1024, 1024]", permute_1029: "f32[1024, 1024]", div_120: "f32[1, 512, 1]", permute_1033: "f32[1024, 4096]", permute_1037: "f32[4096, 1024]", div_121: "f32[1, 512, 1]", permute_1041: "f32[1024, 1024]", permute_1053: "f32[1024, 1024]", permute_1058: "f32[1024, 1024]", permute_1062: "f32[1024, 1024]", div_123: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 29056]"):
    # No stacktrace found for following nodes
    convert_element_type_default_23: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None
    mul_tensor_92: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_23, 1.1111111111111112);  convert_element_type_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_22: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_291, torch.float32);  getitem_291 = None
    mul_tensor_88: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_22, 1.1111111111111112);  convert_element_type_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_21: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_289, torch.float32);  getitem_289 = None
    mul_tensor_84: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_21, 1.1111111111111112);  convert_element_type_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_20: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None
    mul_tensor_80: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_20, 1.1111111111111112);  convert_element_type_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_19: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_285, torch.float32);  getitem_285 = None
    mul_tensor_76: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_19, 1.1111111111111112);  convert_element_type_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_18: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_283, torch.float32);  getitem_283 = None
    mul_tensor_72: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, 1.1111111111111112);  convert_element_type_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_17: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
    mul_tensor_68: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_17, 1.1111111111111112);  convert_element_type_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_16: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_279, torch.float32);  getitem_279 = None
    mul_tensor_64: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_16, 1.1111111111111112);  convert_element_type_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_15: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_277, torch.float32);  getitem_277 = None
    mul_tensor_60: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1.1111111111111112);  convert_element_type_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_14: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None
    mul_tensor_56: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_14, 1.1111111111111112);  convert_element_type_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_13: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None
    mul_tensor_52: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1.1111111111111112);  convert_element_type_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_12: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_271, torch.float32);  getitem_271 = None
    mul_tensor_48: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_12, 1.1111111111111112);  convert_element_type_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_11: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
    mul_tensor_44: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.1111111111111112);  convert_element_type_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_76, [1, 512, 4096]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_90: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
    erf_12: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_104: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_10: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_267, torch.float32);  getitem_267 = None
    mul_tensor_40: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1.1111111111111112);  convert_element_type_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_82, [1, 512, 4096]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_97: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
    erf_13: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_112: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_9: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None
    mul_tensor_36: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_327: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_88, [1, 512, 4096]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_14: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_120: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_8: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
    mul_tensor_32: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_349: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_94, [1, 512, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_111: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
    erf_15: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_128: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_7: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_261, torch.float32);  getitem_261 = None
    mul_tensor_28: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.1111111111111112);  convert_element_type_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_371: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_100, [1, 512, 4096]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_16: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_136: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_6: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_259, torch.float32);  getitem_259 = None
    mul_tensor_24: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_393: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_106, [1, 512, 4096]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_125: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_17: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
    add_144: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_5: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
    mul_tensor_20: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.1111111111111112);  convert_element_type_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_415: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_112, [1, 512, 4096]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_132: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_18: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_152: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_4: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
    mul_tensor_16: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_118, [1, 512, 4096]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_139: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
    erf_19: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_160: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_3: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_253, torch.float32);  getitem_253 = None
    mul_tensor_12: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1.1111111111111112);  convert_element_type_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_459: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_124, [1, 512, 4096]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_146: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
    erf_20: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_168: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_2: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None
    mul_tensor_8: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_130, [1, 512, 4096]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_153: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
    erf_21: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_176: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_1: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
    mul_tensor_4: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_503: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_136, [1, 512, 4096]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_160: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
    erf_22: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_184: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_247, torch.float32);  getitem_247 = None
    mul_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_525: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(addmm_142, [1, 512, 4096]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
    erf_23: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_192: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:646, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, [1, 512, 1024]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_172: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
    erf_24: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_196: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1233, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_49: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    full_default_5: "f32[511, 29056]" = torch.ops.aten.full.default([511, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[511, 29056]" = torch.ops.aten.scatter.value(full_default_5, 1, where_2, -1.0);  full_default_5 = where_2 = None
    where_3: "f32[511, 1]" = torch.ops.aten.where.self(ne_3, div_49, full_default_3);  ne_3 = div_49 = None
    mul_176: "f32[511, 29056]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_25: "f32[511, 29056]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_28: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [1], True)
    mul_177: "f32[511, 29056]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_77: "f32[511, 29056]" = torch.ops.aten.sub.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    view_534: "f32[1, 511, 29056]" = torch.ops.aten.reshape.default(sub_77, [1, 511, 29056]);  sub_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1230, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    full_default_7: "f32[1, 511, 29056]" = torch.ops.aten.full.default([1, 511, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_8: "f32[1, 512, 29056]" = torch.ops.aten.full.default([1, 512, 29056], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 29056]" = torch.ops.aten.slice_scatter.default(full_default_8, view_534, 1, 0, -1);  full_default_8 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1230, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    add_199: "f32[1, 512, 29056]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:669, code: hidden_states = self.decoder(hidden_states)
    view_535: "f32[512, 29056]" = torch.ops.aten.reshape.default(add_199, [512, 29056]);  add_199 = None
    mm: "f32[512, 1024]" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
    permute_267: "f32[29056, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_1: "f32[29056, 1024]" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
    permute_268: "f32[1024, 29056]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_29: "f32[1, 29056]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[29056]" = torch.ops.aten.reshape.default(sum_29, [29056]);  sum_29 = None
    permute_269: "f32[29056, 1024]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_537: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm, [1, 512, 1024]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:648, code: hidden_states = self.LayerNorm(hidden_states)
    mul_179: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, primals_392);  primals_392 = None
    mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_179, 1024)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_179, mul_174);  mul_179 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_174, sum_31);  sum_31 = None
    sub_79: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_180, sum_30);  mul_180 = sum_30 = None
    sub_80: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_79, mul_182);  sub_79 = mul_182 = None
    mul_183: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_80);  div_50 = sub_80 = None
    mul_184: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, mul_174);  mul_174 = None
    sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_186: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_187: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, view_529)
    mul_188: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_187, -0.5);  mul_187 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_188);  mul_188 = None
    mul_189: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_190: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, mul_189);  view_529 = mul_189 = None
    add_201: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_186, mul_190);  mul_186 = mul_190 = None
    mul_191: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_183, add_201);  mul_183 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:646, code: hidden_states = self.dense(hidden_states)
    view_538: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_191, [512, 1024]);  mul_191 = None
    mm_2: "f32[512, 1024]" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
    permute_271: "f32[1024, 512]" = torch.ops.aten.permute.default(view_538, [1, 0])
    mm_3: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
    permute_272: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_34: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
    view_539: "f32[1024]" = torch.ops.aten.reshape.default(sum_34, [1024]);  sum_34 = None
    permute_273: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_540: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, primals_388);  primals_388 = None
    mul_194: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_193, 1024)
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
    mul_195: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_193, mul_169);  mul_193 = None
    sum_36: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    mul_196: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_169, sum_36);  sum_36 = None
    sub_82: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_194, sum_35);  mul_194 = sum_35 = None
    sub_83: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_82, mul_196);  sub_82 = mul_196 = None
    mul_197: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_83);  div_51 = sub_83 = None
    mul_198: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, mul_169);  mul_169 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
    sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None
    mul_199: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_200: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_197, mul_199);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_541: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_200, [512, 1024]);  mul_200 = None
    mm_4: "f32[512, 4096]" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
    permute_275: "f32[1024, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_5: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
    permute_276: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_39: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[1024]" = torch.ops.aten.reshape.default(sum_39, [1024]);  sum_39 = None
    permute_277: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_543: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_202: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
    mul_203: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, view_525)
    mul_204: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
    exp_27: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_204);  mul_204 = None
    mul_205: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_206: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, mul_205);  view_525 = mul_205 = None
    add_203: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
    mul_207: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_543, add_203);  view_543 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_544: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_207, [512, 4096]);  mul_207 = None
    mm_6: "f32[512, 1024]" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
    permute_279: "f32[4096, 512]" = torch.ops.aten.permute.default(view_544, [1, 0])
    mm_7: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
    permute_280: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_40: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
    view_545: "f32[4096]" = torch.ops.aten.reshape.default(sum_40, [4096]);  sum_40 = None
    permute_281: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_546: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_209: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, primals_382);  primals_382 = None
    mul_210: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_209, 1024)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
    mul_211: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_209, mul_164);  mul_209 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    mul_212: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_164, sum_42);  sum_42 = None
    sub_85: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_210, sum_41);  mul_210 = sum_41 = None
    sub_86: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_85, mul_212);  sub_85 = mul_212 = None
    mul_213: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_86);  div_52 = sub_86 = None
    mul_214: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, mul_164);  mul_164 = None
    sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_546, [0, 1]);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_204: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_197, mul_213);  mul_197 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
    mul_215: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_216: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_204, mul_215);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_547: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_216, [512, 1024]);  mul_216 = None
    mm_8: "f32[512, 1024]" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
    permute_283: "f32[1024, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
    permute_284: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_45: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[1024]" = torch.ops.aten.reshape.default(sum_45, [1024]);  sum_45 = None
    permute_285: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_549: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_550: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_549, [1, 512, 16, 64]);  view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
    
    # No stacktrace found for following nodes
    view_default_6: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_286, [16, 512, 64]);  permute_286 = None
    bmm_default_2: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_1, view_default_6);  permute_default_1 = None
    view_default_7: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_2, [1, 16, 512, 64]);  bmm_default_2 = None
    bmm_default_3: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
    view_default_8: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_3, [1, 16, 512, 512]);  bmm_default_3 = None
    mul_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor);  view_default_8 = mul_tensor = None
    mul_tensor_2: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, alias_default_1);  mul_tensor_1 = None
    sum_dim_int_list_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    mul_tensor_3: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_1, sum_dim_int_list_1);  alias_default_1 = sum_dim_int_list_1 = None
    sub_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
    view_default_9: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_1, [16, 512, 512]);  sub_tensor_1 = None
    bmm_default_4: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_3, view_default_9);  permute_default_3 = None
    view_default_10: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_4, [1, 16, 64, 512]);  bmm_default_4 = None
    mul_scalar_2: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_10, 0.3535533905932738);  view_default_10 = None
    permute_default_5: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    bmm_default_5: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_9, permute_default_4);  view_default_9 = permute_default_4 = None
    view_default_11: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_5, [1, 16, 512, 64]);  bmm_default_5 = None
    mul_scalar_3: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_3, [0, 2, 1, 3]);  mul_scalar_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_557: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_27, [1, 512, 1024]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_28: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_558: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_28, [1, 512, 1024]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_559: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_558, [512, 1024]);  view_558 = None
    mm_10: "f32[512, 1024]" = torch.ops.aten.mm.default(view_559, permute_294);  permute_294 = None
    permute_295: "f32[1024, 512]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_295, view_506);  permute_295 = None
    permute_296: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[1024]" = torch.ops.aten.reshape.default(sum_47, [1024]);  sum_47 = None
    permute_297: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_561: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_5, [0, 2, 1, 3]);  permute_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_562: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_298, [1, 512, 1024]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_563: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_562, [512, 1024]);  view_562 = None
    mm_12: "f32[512, 1024]" = torch.ops.aten.mm.default(view_563, permute_299);  permute_299 = None
    permute_300: "f32[1024, 512]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_300, view_506);  permute_300 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[1024]" = torch.ops.aten.reshape.default(sum_48, [1024]);  sum_48 = None
    permute_302: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_565: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_205: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_561, view_565);  view_561 = view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_566: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_557, [512, 1024]);  view_557 = None
    mm_14: "f32[512, 1024]" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
    permute_304: "f32[1024, 512]" = torch.ops.aten.permute.default(view_566, [1, 0])
    mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
    permute_305: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_49: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
    view_567: "f32[1024]" = torch.ops.aten.reshape.default(sum_49, [1024]);  sum_49 = None
    permute_306: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_568: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_206: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_205, view_568);  add_205 = view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, primals_372);  primals_372 = None
    mul_223: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_222, 1024)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_222, mul_162);  mul_222 = None
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, sum_51);  sum_51 = None
    sub_89: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_223, sum_50);  mul_223 = sum_50 = None
    sub_90: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_89, mul_225);  sub_89 = mul_225 = None
    mul_226: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_90);  div_54 = sub_90 = None
    mul_227: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, mul_162);  mul_162 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_207: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_204, mul_226);  add_204 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
    mul_228: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_229: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_207, mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_569: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_229, [512, 1024]);  mul_229 = None
    mm_16: "f32[512, 4096]" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
    permute_308: "f32[1024, 512]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_17: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
    permute_309: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_54: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[1024]" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
    permute_310: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_571: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 4096]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_231: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_232: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, view_503)
    mul_233: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
    exp_28: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
    mul_234: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_235: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, mul_234);  view_503 = mul_234 = None
    add_209: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
    mul_236: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_571, add_209);  view_571 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_572: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_236, [512, 4096]);  mul_236 = None
    mm_18: "f32[512, 1024]" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
    permute_312: "f32[4096, 512]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_19: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
    permute_313: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_55: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[4096]" = torch.ops.aten.reshape.default(sum_55, [4096]);  sum_55 = None
    permute_314: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_574: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_238: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, primals_366);  primals_366 = None
    mul_239: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_238, 1024)
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_238, mul_157);  mul_238 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, sum_57);  sum_57 = None
    sub_92: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_239, sum_56);  mul_239 = sum_56 = None
    sub_93: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_92, mul_241);  sub_92 = mul_241 = None
    mul_242: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_93);  div_55 = sub_93 = None
    mul_243: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, mul_157);  mul_157 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_210: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_207, mul_242);  add_207 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None
    mul_244: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_245: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_210, mul_244);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_575: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_245, [512, 1024]);  mul_245 = None
    mm_20: "f32[512, 1024]" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
    permute_316: "f32[1024, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_21: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
    permute_317: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[1024]" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
    permute_318: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_577: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 1024]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_578: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_577, [1, 512, 16, 64]);  view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
    
    # No stacktrace found for following nodes
    view_default_18: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_319, [16, 512, 64]);  permute_319 = None
    bmm_default_8: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_7, view_default_18);  permute_default_7 = None
    view_default_19: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_8, [1, 16, 512, 64]);  bmm_default_8 = None
    bmm_default_9: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_18, permute_default_8);  view_default_18 = permute_default_8 = None
    view_default_20: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_9, [1, 16, 512, 512]);  bmm_default_9 = None
    mul_tensor_5: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_20, mul_tensor_4);  view_default_20 = mul_tensor_4 = None
    mul_tensor_6: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_5, alias_default_3);  mul_tensor_5 = None
    sum_dim_int_list_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [-1], True)
    mul_tensor_7: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_3, sum_dim_int_list_3);  alias_default_3 = sum_dim_int_list_3 = None
    sub_tensor_3: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
    view_default_21: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_3, [16, 512, 512]);  sub_tensor_3 = None
    bmm_default_10: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_9, view_default_21);  permute_default_9 = None
    view_default_22: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_10, [1, 16, 64, 512]);  bmm_default_10 = None
    mul_scalar_6: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_22, 0.3535533905932738);  view_default_22 = None
    permute_default_11: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_6, [0, 1, 3, 2]);  mul_scalar_6 = None
    bmm_default_11: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_21, permute_default_10);  view_default_21 = permute_default_10 = None
    view_default_23: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_11, [1, 16, 512, 64]);  bmm_default_11 = None
    mul_scalar_7: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_23, 0.3535533905932738);  view_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_7, [0, 2, 1, 3]);  mul_scalar_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_585: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_32, [1, 512, 1024]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_19, [0, 2, 1, 3]);  view_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_33: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_586: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_33, [1, 512, 1024]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_587: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_586, [512, 1024]);  view_586 = None
    mm_22: "f32[512, 1024]" = torch.ops.aten.mm.default(view_587, permute_327);  permute_327 = None
    permute_328: "f32[1024, 512]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_23: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_328, view_484);  permute_328 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[1024]" = torch.ops.aten.reshape.default(sum_62, [1024]);  sum_62 = None
    permute_330: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_589: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 1024]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_590: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_331, [1, 512, 1024]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_591: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_590, [512, 1024]);  view_590 = None
    mm_24: "f32[512, 1024]" = torch.ops.aten.mm.default(view_591, permute_332);  permute_332 = None
    permute_333: "f32[1024, 512]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_333, view_484);  permute_333 = None
    permute_334: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[1024]" = torch.ops.aten.reshape.default(sum_63, [1024]);  sum_63 = None
    permute_335: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_593: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 1024]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_211: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_589, view_593);  view_589 = view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_594: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_585, [512, 1024]);  view_585 = None
    mm_26: "f32[512, 1024]" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
    permute_337: "f32[1024, 512]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_27: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
    permute_338: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_64: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[1024]" = torch.ops.aten.reshape.default(sum_64, [1024]);  sum_64 = None
    permute_339: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_596: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_212: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_211, view_596);  add_211 = view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_251: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, primals_356);  primals_356 = None
    mul_252: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_251, 1024)
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_251, mul_155);  mul_251 = None
    sum_66: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_155, sum_66);  sum_66 = None
    sub_96: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_252, sum_65);  mul_252 = sum_65 = None
    sub_97: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_96, mul_254);  sub_96 = mul_254 = None
    mul_255: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_97);  div_57 = sub_97 = None
    mul_256: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, mul_155);  mul_155 = None
    sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_213: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_210, mul_255);  add_210 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
    mul_257: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_258: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_213, mul_257);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_597: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_258, [512, 1024]);  mul_258 = None
    mm_28: "f32[512, 4096]" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
    permute_341: "f32[1024, 512]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_29: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
    permute_342: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_69: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[1024]" = torch.ops.aten.reshape.default(sum_69, [1024]);  sum_69 = None
    permute_343: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_599: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_260: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_261: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, view_481)
    mul_262: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
    exp_29: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
    mul_263: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_264: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, mul_263);  view_481 = mul_263 = None
    add_215: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
    mul_265: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_599, add_215);  view_599 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_600: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_265, [512, 4096]);  mul_265 = None
    mm_30: "f32[512, 1024]" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
    permute_345: "f32[4096, 512]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_31: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
    permute_346: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[4096]" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
    permute_347: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_602: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 1024]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_267: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, primals_350);  primals_350 = None
    mul_268: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_267, 1024)
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_267, mul_150);  mul_267 = None
    sum_72: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, sum_72);  sum_72 = None
    sub_99: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_268, sum_71);  mul_268 = sum_71 = None
    sub_100: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_99, mul_270);  sub_99 = mul_270 = None
    mul_271: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_100);  div_58 = sub_100 = None
    mul_272: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, mul_150);  mul_150 = None
    sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_602, [0, 1]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_216: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_213, mul_271);  add_213 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None
    mul_273: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_274: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_216, mul_273);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_603: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_274, [512, 1024]);  mul_274 = None
    mm_32: "f32[512, 1024]" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
    permute_349: "f32[1024, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
    permute_350: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[1024]" = torch.ops.aten.reshape.default(sum_75, [1024]);  sum_75 = None
    permute_351: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_605: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_606: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_605, [1, 512, 16, 64]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
    
    # No stacktrace found for following nodes
    view_default_30: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_352, [16, 512, 64]);  permute_352 = None
    bmm_default_14: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_13, view_default_30);  permute_default_13 = None
    view_default_31: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_14, [1, 16, 512, 64]);  bmm_default_14 = None
    bmm_default_15: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_30, permute_default_14);  view_default_30 = permute_default_14 = None
    view_default_32: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_15, [1, 16, 512, 512]);  bmm_default_15 = None
    mul_tensor_9: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_32, mul_tensor_8);  view_default_32 = mul_tensor_8 = None
    mul_tensor_10: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_9, alias_default_5);  mul_tensor_9 = None
    sum_dim_int_list_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [-1], True)
    mul_tensor_11: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_5, sum_dim_int_list_5);  alias_default_5 = sum_dim_int_list_5 = None
    sub_tensor_5: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
    view_default_33: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_5, [16, 512, 512]);  sub_tensor_5 = None
    bmm_default_16: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_15, view_default_33);  permute_default_15 = None
    view_default_34: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_16, [1, 16, 64, 512]);  bmm_default_16 = None
    mul_scalar_10: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_34, 0.3535533905932738);  view_default_34 = None
    permute_default_17: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_10, [0, 1, 3, 2]);  mul_scalar_10 = None
    bmm_default_17: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_33, permute_default_16);  view_default_33 = permute_default_16 = None
    view_default_35: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_17, [1, 16, 512, 64]);  bmm_default_17 = None
    mul_scalar_11: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_35, 0.3535533905932738);  view_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_11, [0, 2, 1, 3]);  mul_scalar_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_613: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_37, [1, 512, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_31, [0, 2, 1, 3]);  view_default_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_38: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_614: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_38, [1, 512, 1024]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_615: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_614, [512, 1024]);  view_614 = None
    mm_34: "f32[512, 1024]" = torch.ops.aten.mm.default(view_615, permute_360);  permute_360 = None
    permute_361: "f32[1024, 512]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_35: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_361, view_462);  permute_361 = None
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[1024]" = torch.ops.aten.reshape.default(sum_77, [1024]);  sum_77 = None
    permute_363: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_617: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_17, [0, 2, 1, 3]);  permute_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_618: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_364, [1, 512, 1024]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_619: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_618, [512, 1024]);  view_618 = None
    mm_36: "f32[512, 1024]" = torch.ops.aten.mm.default(view_619, permute_365);  permute_365 = None
    permute_366: "f32[1024, 512]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_37: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_366, view_462);  permute_366 = None
    permute_367: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
    view_620: "f32[1024]" = torch.ops.aten.reshape.default(sum_78, [1024]);  sum_78 = None
    permute_368: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_621: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 1024]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_217: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_617, view_621);  view_617 = view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_622: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_613, [512, 1024]);  view_613 = None
    mm_38: "f32[512, 1024]" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
    permute_370: "f32[1024, 512]" = torch.ops.aten.permute.default(view_622, [1, 0])
    mm_39: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
    permute_371: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_79: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
    view_623: "f32[1024]" = torch.ops.aten.reshape.default(sum_79, [1024]);  sum_79 = None
    permute_372: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_624: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 1024]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_218: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_217, view_624);  add_217 = view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_280: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, primals_340);  primals_340 = None
    mul_281: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, 1024)
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, mul_148);  mul_280 = None
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, sum_81);  sum_81 = None
    sub_103: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_281, sum_80);  mul_281 = sum_80 = None
    sub_104: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_103, mul_283);  sub_103 = mul_283 = None
    mul_284: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_104);  div_60 = sub_104 = None
    mul_285: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, mul_148);  mul_148 = None
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_219: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_216, mul_284);  add_216 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None
    mul_286: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_287: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_219, mul_286);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_625: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_287, [512, 1024]);  mul_287 = None
    mm_40: "f32[512, 4096]" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
    permute_374: "f32[1024, 512]" = torch.ops.aten.permute.default(view_625, [1, 0])
    mm_41: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
    permute_375: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
    view_626: "f32[1024]" = torch.ops.aten.reshape.default(sum_84, [1024]);  sum_84 = None
    permute_376: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_627: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 4096]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_289: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
    mul_290: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, view_459)
    mul_291: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_30: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_293: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, mul_292);  view_459 = mul_292 = None
    add_221: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_627, add_221);  view_627 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_294, [512, 4096]);  mul_294 = None
    mm_42: "f32[512, 1024]" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
    permute_378: "f32[4096, 512]" = torch.ops.aten.permute.default(view_628, [1, 0])
    mm_43: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
    permute_379: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_85: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
    view_629: "f32[4096]" = torch.ops.aten.reshape.default(sum_85, [4096]);  sum_85 = None
    permute_380: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_630: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 1024]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_296: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, primals_334);  primals_334 = None
    mul_297: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_296, 1024)
    sum_86: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
    mul_298: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_296, mul_143);  mul_296 = None
    sum_87: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_143, sum_87);  sum_87 = None
    sub_106: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_297, sum_86);  mul_297 = sum_86 = None
    sub_107: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_106, mul_299);  sub_106 = mul_299 = None
    mul_300: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_107);  div_61 = sub_107 = None
    mul_301: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, mul_143);  mul_143 = None
    sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
    sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_630, [0, 1]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_222: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_219, mul_300);  add_219 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
    mul_302: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_303: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_222, mul_302);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_631: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_303, [512, 1024]);  mul_303 = None
    mm_44: "f32[512, 1024]" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
    permute_382: "f32[1024, 512]" = torch.ops.aten.permute.default(view_631, [1, 0])
    mm_45: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
    permute_383: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
    view_632: "f32[1024]" = torch.ops.aten.reshape.default(sum_90, [1024]);  sum_90 = None
    permute_384: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_633: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 1024]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_634: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_633, [1, 512, 16, 64]);  view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
    
    # No stacktrace found for following nodes
    view_default_42: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_385, [16, 512, 64]);  permute_385 = None
    bmm_default_20: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_19, view_default_42);  permute_default_19 = None
    view_default_43: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_20, [1, 16, 512, 64]);  bmm_default_20 = None
    bmm_default_21: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_42, permute_default_20);  view_default_42 = permute_default_20 = None
    view_default_44: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_21, [1, 16, 512, 512]);  bmm_default_21 = None
    mul_tensor_13: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_44, mul_tensor_12);  view_default_44 = mul_tensor_12 = None
    mul_tensor_14: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_13, alias_default_7);  mul_tensor_13 = None
    sum_dim_int_list_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [-1], True)
    mul_tensor_15: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_7, sum_dim_int_list_7);  alias_default_7 = sum_dim_int_list_7 = None
    sub_tensor_7: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
    view_default_45: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_7, [16, 512, 512]);  sub_tensor_7 = None
    bmm_default_22: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_21, view_default_45);  permute_default_21 = None
    view_default_46: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_22, [1, 16, 64, 512]);  bmm_default_22 = None
    mul_scalar_14: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_46, 0.3535533905932738);  view_default_46 = None
    permute_default_23: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_14, [0, 1, 3, 2]);  mul_scalar_14 = None
    bmm_default_23: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_45, permute_default_22);  view_default_45 = permute_default_22 = None
    view_default_47: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_23, [1, 16, 512, 64]);  bmm_default_23 = None
    mul_scalar_15: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_47, 0.3535533905932738);  view_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_15, [0, 2, 1, 3]);  mul_scalar_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_641: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_42, [1, 512, 1024]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_43, [0, 2, 1, 3]);  view_default_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_642: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_43, [1, 512, 1024]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_643: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_642, [512, 1024]);  view_642 = None
    mm_46: "f32[512, 1024]" = torch.ops.aten.mm.default(view_643, permute_393);  permute_393 = None
    permute_394: "f32[1024, 512]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_47: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_394, view_440);  permute_394 = None
    permute_395: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
    view_644: "f32[1024]" = torch.ops.aten.reshape.default(sum_92, [1024]);  sum_92 = None
    permute_396: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_645: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 1024]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_23, [0, 2, 1, 3]);  permute_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_646: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_397, [1, 512, 1024]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_647: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_646, [512, 1024]);  view_646 = None
    mm_48: "f32[512, 1024]" = torch.ops.aten.mm.default(view_647, permute_398);  permute_398 = None
    permute_399: "f32[1024, 512]" = torch.ops.aten.permute.default(view_647, [1, 0])
    mm_49: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_399, view_440);  permute_399 = None
    permute_400: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_647, [0], True);  view_647 = None
    view_648: "f32[1024]" = torch.ops.aten.reshape.default(sum_93, [1024]);  sum_93 = None
    permute_401: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_649: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 1024]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_223: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_645, view_649);  view_645 = view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_650: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_641, [512, 1024]);  view_641 = None
    mm_50: "f32[512, 1024]" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
    permute_403: "f32[1024, 512]" = torch.ops.aten.permute.default(view_650, [1, 0])
    mm_51: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
    permute_404: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_94: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[1024]" = torch.ops.aten.reshape.default(sum_94, [1024]);  sum_94 = None
    permute_405: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_652: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 1024]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_224: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_223, view_652);  add_223 = view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_309: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, primals_324);  primals_324 = None
    mul_310: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_309, 1024)
    sum_95: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_309, mul_141);  mul_309 = None
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_141, sum_96);  sum_96 = None
    sub_110: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_310, sum_95);  mul_310 = sum_95 = None
    sub_111: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_110, mul_312);  sub_110 = mul_312 = None
    mul_313: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_111);  div_63 = sub_111 = None
    mul_314: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_141);  mul_141 = None
    sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_225: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_222, mul_313);  add_222 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
    mul_315: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_316: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_225, mul_315);  mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_653: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_316, [512, 1024]);  mul_316 = None
    mm_52: "f32[512, 4096]" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
    permute_407: "f32[1024, 512]" = torch.ops.aten.permute.default(view_653, [1, 0])
    mm_53: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
    permute_408: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
    view_654: "f32[1024]" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
    permute_409: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_655: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 4096]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_318: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_319: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, view_437)
    mul_320: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
    exp_31: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
    mul_321: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_322: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, mul_321);  view_437 = mul_321 = None
    add_227: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
    mul_323: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_655, add_227);  view_655 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_656: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_323, [512, 4096]);  mul_323 = None
    mm_54: "f32[512, 1024]" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
    permute_411: "f32[4096, 512]" = torch.ops.aten.permute.default(view_656, [1, 0])
    mm_55: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
    permute_412: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
    view_657: "f32[4096]" = torch.ops.aten.reshape.default(sum_100, [4096]);  sum_100 = None
    permute_413: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_658: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_54, [1, 512, 1024]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_325: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, primals_318);  primals_318 = None
    mul_326: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, 1024)
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, mul_136);  mul_325 = None
    sum_102: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_136, sum_102);  sum_102 = None
    sub_113: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_326, sum_101);  mul_326 = sum_101 = None
    sub_114: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_113, mul_328);  sub_113 = mul_328 = None
    mul_329: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_114);  div_64 = sub_114 = None
    mul_330: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, mul_136);  mul_136 = None
    sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_228: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_329);  add_225 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
    mul_331: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_332: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_228, mul_331);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_659: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_332, [512, 1024]);  mul_332 = None
    mm_56: "f32[512, 1024]" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
    permute_415: "f32[1024, 512]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_57: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
    permute_416: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_105: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
    view_660: "f32[1024]" = torch.ops.aten.reshape.default(sum_105, [1024]);  sum_105 = None
    permute_417: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_661: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_56, [1, 512, 1024]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_662: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_661, [1, 512, 16, 64]);  view_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None
    
    # No stacktrace found for following nodes
    view_default_54: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_418, [16, 512, 64]);  permute_418 = None
    bmm_default_26: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_25, view_default_54);  permute_default_25 = None
    view_default_55: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_26, [1, 16, 512, 64]);  bmm_default_26 = None
    bmm_default_27: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_54, permute_default_26);  view_default_54 = permute_default_26 = None
    view_default_56: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_27, [1, 16, 512, 512]);  bmm_default_27 = None
    mul_tensor_17: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_56, mul_tensor_16);  view_default_56 = mul_tensor_16 = None
    mul_tensor_18: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_17, alias_default_9);  mul_tensor_17 = None
    sum_dim_int_list_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [-1], True)
    mul_tensor_19: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_9, sum_dim_int_list_9);  alias_default_9 = sum_dim_int_list_9 = None
    sub_tensor_9: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
    view_default_57: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_9, [16, 512, 512]);  sub_tensor_9 = None
    bmm_default_28: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_27, view_default_57);  permute_default_27 = None
    view_default_58: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_28, [1, 16, 64, 512]);  bmm_default_28 = None
    mul_scalar_18: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_58, 0.3535533905932738);  view_default_58 = None
    permute_default_29: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_18, [0, 1, 3, 2]);  mul_scalar_18 = None
    bmm_default_29: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_57, permute_default_28);  view_default_57 = permute_default_28 = None
    view_default_59: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_29, [1, 16, 512, 64]);  bmm_default_29 = None
    mul_scalar_19: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_59, 0.3535533905932738);  view_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_19, [0, 2, 1, 3]);  mul_scalar_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_669: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_47, [1, 512, 1024]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_55, [0, 2, 1, 3]);  view_default_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_670: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_48, [1, 512, 1024]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_671: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_670, [512, 1024]);  view_670 = None
    mm_58: "f32[512, 1024]" = torch.ops.aten.mm.default(view_671, permute_426);  permute_426 = None
    permute_427: "f32[1024, 512]" = torch.ops.aten.permute.default(view_671, [1, 0])
    mm_59: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_427, view_418);  permute_427 = None
    permute_428: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_671, [0], True);  view_671 = None
    view_672: "f32[1024]" = torch.ops.aten.reshape.default(sum_107, [1024]);  sum_107 = None
    permute_429: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_673: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_58, [1, 512, 1024]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_29, [0, 2, 1, 3]);  permute_default_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_674: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_430, [1, 512, 1024]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_675: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_674, [512, 1024]);  view_674 = None
    mm_60: "f32[512, 1024]" = torch.ops.aten.mm.default(view_675, permute_431);  permute_431 = None
    permute_432: "f32[1024, 512]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_61: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_432, view_418);  permute_432 = None
    permute_433: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[1024]" = torch.ops.aten.reshape.default(sum_108, [1024]);  sum_108 = None
    permute_434: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_677: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_60, [1, 512, 1024]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_229: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_673, view_677);  view_673 = view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_678: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_669, [512, 1024]);  view_669 = None
    mm_62: "f32[512, 1024]" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
    permute_436: "f32[1024, 512]" = torch.ops.aten.permute.default(view_678, [1, 0])
    mm_63: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
    permute_437: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_109: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
    view_679: "f32[1024]" = torch.ops.aten.reshape.default(sum_109, [1024]);  sum_109 = None
    permute_438: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_680: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_62, [1, 512, 1024]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_230: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_229, view_680);  add_229 = view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_338: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, primals_308);  primals_308 = None
    mul_339: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_338, 1024)
    sum_110: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_338, mul_134);  mul_338 = None
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_134, sum_111);  sum_111 = None
    sub_117: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_339, sum_110);  mul_339 = sum_110 = None
    sub_118: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_117, mul_341);  sub_117 = mul_341 = None
    mul_342: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_118);  div_66 = sub_118 = None
    mul_343: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, mul_134);  mul_134 = None
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_231: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_228, mul_342);  add_228 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
    mul_344: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_345: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_231, mul_344);  mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_681: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_345, [512, 1024]);  mul_345 = None
    mm_64: "f32[512, 4096]" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
    permute_440: "f32[1024, 512]" = torch.ops.aten.permute.default(view_681, [1, 0])
    mm_65: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
    permute_441: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_114: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
    view_682: "f32[1024]" = torch.ops.aten.reshape.default(sum_114, [1024]);  sum_114 = None
    permute_442: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_683: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_64, [1, 512, 4096]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_347: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_348: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_349: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
    exp_32: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_349);  mul_349 = None
    mul_350: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_351: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, mul_350);  view_415 = mul_350 = None
    add_233: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
    mul_352: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_683, add_233);  view_683 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_684: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_352, [512, 4096]);  mul_352 = None
    mm_66: "f32[512, 1024]" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
    permute_444: "f32[4096, 512]" = torch.ops.aten.permute.default(view_684, [1, 0])
    mm_67: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
    permute_445: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
    view_685: "f32[4096]" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
    permute_446: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_686: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_66, [1, 512, 1024]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_354: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, primals_302);  primals_302 = None
    mul_355: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_354, 1024)
    sum_116: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_354, mul_129);  mul_354 = None
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_129, sum_117);  sum_117 = None
    sub_120: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_355, sum_116);  mul_355 = sum_116 = None
    sub_121: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_120, mul_357);  sub_120 = mul_357 = None
    mul_358: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_121);  div_67 = sub_121 = None
    mul_359: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, mul_129);  mul_129 = None
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_686, [0, 1]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_234: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_231, mul_358);  add_231 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None
    mul_360: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_361: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_234, mul_360);  mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_687: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_361, [512, 1024]);  mul_361 = None
    mm_68: "f32[512, 1024]" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
    permute_448: "f32[1024, 512]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_69: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
    permute_449: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_120: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
    view_688: "f32[1024]" = torch.ops.aten.reshape.default(sum_120, [1024]);  sum_120 = None
    permute_450: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_689: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_68, [1, 512, 1024]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_690: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_689, [1, 512, 16, 64]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
    
    # No stacktrace found for following nodes
    view_default_66: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_451, [16, 512, 64]);  permute_451 = None
    bmm_default_32: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_31, view_default_66);  permute_default_31 = None
    view_default_67: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_32, [1, 16, 512, 64]);  bmm_default_32 = None
    bmm_default_33: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_66, permute_default_32);  view_default_66 = permute_default_32 = None
    view_default_68: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_33, [1, 16, 512, 512]);  bmm_default_33 = None
    mul_tensor_21: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_68, mul_tensor_20);  view_default_68 = mul_tensor_20 = None
    mul_tensor_22: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_21, alias_default_11);  mul_tensor_21 = None
    sum_dim_int_list_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [-1], True)
    mul_tensor_23: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_11, sum_dim_int_list_11);  alias_default_11 = sum_dim_int_list_11 = None
    sub_tensor_11: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
    view_default_69: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_11, [16, 512, 512]);  sub_tensor_11 = None
    bmm_default_34: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_33, view_default_69);  permute_default_33 = None
    view_default_70: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_34, [1, 16, 64, 512]);  bmm_default_34 = None
    mul_scalar_22: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_70, 0.3535533905932738);  view_default_70 = None
    permute_default_35: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_22, [0, 1, 3, 2]);  mul_scalar_22 = None
    bmm_default_35: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_69, permute_default_34);  view_default_69 = permute_default_34 = None
    view_default_71: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_35, [1, 16, 512, 64]);  bmm_default_35 = None
    mul_scalar_23: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_71, 0.3535533905932738);  view_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_23, [0, 2, 1, 3]);  mul_scalar_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_697: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_52, [1, 512, 1024]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_67, [0, 2, 1, 3]);  view_default_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_698: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_53, [1, 512, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_699: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_698, [512, 1024]);  view_698 = None
    mm_70: "f32[512, 1024]" = torch.ops.aten.mm.default(view_699, permute_459);  permute_459 = None
    permute_460: "f32[1024, 512]" = torch.ops.aten.permute.default(view_699, [1, 0])
    mm_71: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_460, view_396);  permute_460 = None
    permute_461: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_699, [0], True);  view_699 = None
    view_700: "f32[1024]" = torch.ops.aten.reshape.default(sum_122, [1024]);  sum_122 = None
    permute_462: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_701: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_70, [1, 512, 1024]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_702: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_463, [1, 512, 1024]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_703: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_702, [512, 1024]);  view_702 = None
    mm_72: "f32[512, 1024]" = torch.ops.aten.mm.default(view_703, permute_464);  permute_464 = None
    permute_465: "f32[1024, 512]" = torch.ops.aten.permute.default(view_703, [1, 0])
    mm_73: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_465, view_396);  permute_465 = None
    permute_466: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_703, [0], True);  view_703 = None
    view_704: "f32[1024]" = torch.ops.aten.reshape.default(sum_123, [1024]);  sum_123 = None
    permute_467: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_705: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_72, [1, 512, 1024]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_235: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_701, view_705);  view_701 = view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_706: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_697, [512, 1024]);  view_697 = None
    mm_74: "f32[512, 1024]" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
    permute_469: "f32[1024, 512]" = torch.ops.aten.permute.default(view_706, [1, 0])
    mm_75: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
    permute_470: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_124: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
    view_707: "f32[1024]" = torch.ops.aten.reshape.default(sum_124, [1024]);  sum_124 = None
    permute_471: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_708: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_74, [1, 512, 1024]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_236: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_235, view_708);  add_235 = view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_367: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, primals_292);  primals_292 = None
    mul_368: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, 1024)
    sum_125: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, mul_127);  mul_367 = None
    sum_126: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_127, sum_126);  sum_126 = None
    sub_124: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_368, sum_125);  mul_368 = sum_125 = None
    sub_125: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_124, mul_370);  sub_124 = mul_370 = None
    mul_371: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_125);  div_69 = sub_125 = None
    mul_372: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, mul_127);  mul_127 = None
    sum_127: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_237: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_234, mul_371);  add_234 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None
    mul_373: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_374: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_237, mul_373);  mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_709: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_374, [512, 1024]);  mul_374 = None
    mm_76: "f32[512, 4096]" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
    permute_473: "f32[1024, 512]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_77: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
    permute_474: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_129: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
    view_710: "f32[1024]" = torch.ops.aten.reshape.default(sum_129, [1024]);  sum_129 = None
    permute_475: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_711: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_76, [1, 512, 4096]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_376: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_377: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_378: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
    exp_33: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
    mul_379: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_380: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, mul_379);  view_393 = mul_379 = None
    add_239: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
    mul_381: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_711, add_239);  view_711 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_712: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_381, [512, 4096]);  mul_381 = None
    mm_78: "f32[512, 1024]" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
    permute_477: "f32[4096, 512]" = torch.ops.aten.permute.default(view_712, [1, 0])
    mm_79: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
    permute_478: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_130: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
    view_713: "f32[4096]" = torch.ops.aten.reshape.default(sum_130, [4096]);  sum_130 = None
    permute_479: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_714: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_78, [1, 512, 1024]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_383: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, primals_286);  primals_286 = None
    mul_384: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, 1024)
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_122);  mul_383 = None
    sum_132: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, sum_132);  sum_132 = None
    sub_127: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_384, sum_131);  mul_384 = sum_131 = None
    sub_128: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_127, mul_386);  sub_127 = mul_386 = None
    mul_387: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_128);  div_70 = sub_128 = None
    mul_388: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, mul_122);  mul_122 = None
    sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_714, [0, 1]);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_240: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_387);  add_237 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
    mul_389: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_390: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_240, mul_389);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_715: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_390, [512, 1024]);  mul_390 = None
    mm_80: "f32[512, 1024]" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
    permute_481: "f32[1024, 512]" = torch.ops.aten.permute.default(view_715, [1, 0])
    mm_81: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
    permute_482: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_135: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
    view_716: "f32[1024]" = torch.ops.aten.reshape.default(sum_135, [1024]);  sum_135 = None
    permute_483: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_717: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_80, [1, 512, 1024]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_718: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_717, [1, 512, 16, 64]);  view_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
    
    # No stacktrace found for following nodes
    view_default_78: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_484, [16, 512, 64]);  permute_484 = None
    bmm_default_38: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_37, view_default_78);  permute_default_37 = None
    view_default_79: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_38, [1, 16, 512, 64]);  bmm_default_38 = None
    bmm_default_39: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_78, permute_default_38);  view_default_78 = permute_default_38 = None
    view_default_80: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_39, [1, 16, 512, 512]);  bmm_default_39 = None
    mul_tensor_25: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_80, mul_tensor_24);  view_default_80 = mul_tensor_24 = None
    mul_tensor_26: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_25, alias_default_13);  mul_tensor_25 = None
    sum_dim_int_list_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [-1], True)
    mul_tensor_27: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_13, sum_dim_int_list_13);  alias_default_13 = sum_dim_int_list_13 = None
    sub_tensor_13: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
    view_default_81: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_13, [16, 512, 512]);  sub_tensor_13 = None
    bmm_default_40: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_39, view_default_81);  permute_default_39 = None
    view_default_82: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_40, [1, 16, 64, 512]);  bmm_default_40 = None
    mul_scalar_26: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_82, 0.3535533905932738);  view_default_82 = None
    permute_default_41: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_26, [0, 1, 3, 2]);  mul_scalar_26 = None
    bmm_default_41: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_81, permute_default_40);  view_default_81 = permute_default_40 = None
    view_default_83: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_41, [1, 16, 512, 64]);  bmm_default_41 = None
    mul_scalar_27: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_83, 0.3535533905932738);  view_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_27, [0, 2, 1, 3]);  mul_scalar_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_725: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_57, [1, 512, 1024]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_79, [0, 2, 1, 3]);  view_default_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_726: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_58, [1, 512, 1024]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_727: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_726, [512, 1024]);  view_726 = None
    mm_82: "f32[512, 1024]" = torch.ops.aten.mm.default(view_727, permute_492);  permute_492 = None
    permute_493: "f32[1024, 512]" = torch.ops.aten.permute.default(view_727, [1, 0])
    mm_83: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_493, view_374);  permute_493 = None
    permute_494: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_137: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
    view_728: "f32[1024]" = torch.ops.aten.reshape.default(sum_137, [1024]);  sum_137 = None
    permute_495: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_729: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_82, [1, 512, 1024]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_41, [0, 2, 1, 3]);  permute_default_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_730: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_496, [1, 512, 1024]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_731: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_730, [512, 1024]);  view_730 = None
    mm_84: "f32[512, 1024]" = torch.ops.aten.mm.default(view_731, permute_497);  permute_497 = None
    permute_498: "f32[1024, 512]" = torch.ops.aten.permute.default(view_731, [1, 0])
    mm_85: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_498, view_374);  permute_498 = None
    permute_499: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_731, [0], True);  view_731 = None
    view_732: "f32[1024]" = torch.ops.aten.reshape.default(sum_138, [1024]);  sum_138 = None
    permute_500: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_733: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_84, [1, 512, 1024]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_241: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_729, view_733);  view_729 = view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_734: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_725, [512, 1024]);  view_725 = None
    mm_86: "f32[512, 1024]" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
    permute_502: "f32[1024, 512]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_87: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
    permute_503: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_139: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
    view_735: "f32[1024]" = torch.ops.aten.reshape.default(sum_139, [1024]);  sum_139 = None
    permute_504: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_736: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_86, [1, 512, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_242: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_241, view_736);  add_241 = view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_396: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, primals_276);  primals_276 = None
    mul_397: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_396, 1024)
    sum_140: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_396, mul_120);  mul_396 = None
    sum_141: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, sum_141);  sum_141 = None
    sub_131: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_397, sum_140);  mul_397 = sum_140 = None
    sub_132: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_131, mul_399);  sub_131 = mul_399 = None
    mul_400: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_132);  div_72 = sub_132 = None
    mul_401: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, mul_120);  mul_120 = None
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_243: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_240, mul_400);  add_240 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
    mul_402: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_403: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_243, mul_402);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_737: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_403, [512, 1024]);  mul_403 = None
    mm_88: "f32[512, 4096]" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
    permute_506: "f32[1024, 512]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_89: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
    permute_507: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_144: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
    view_738: "f32[1024]" = torch.ops.aten.reshape.default(sum_144, [1024]);  sum_144 = None
    permute_508: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_739: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_88, [1, 512, 4096]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_405: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_406: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_407: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_34: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_409: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, mul_408);  view_371 = mul_408 = None
    add_245: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_739, add_245);  view_739 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_740: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_410, [512, 4096]);  mul_410 = None
    mm_90: "f32[512, 1024]" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
    permute_510: "f32[4096, 512]" = torch.ops.aten.permute.default(view_740, [1, 0])
    mm_91: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
    permute_511: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_145: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
    view_741: "f32[4096]" = torch.ops.aten.reshape.default(sum_145, [4096]);  sum_145 = None
    permute_512: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_742: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_90, [1, 512, 1024]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_412: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, primals_270);  primals_270 = None
    mul_413: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_412, 1024)
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_412, mul_115);  mul_412 = None
    sum_147: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, sum_147);  sum_147 = None
    sub_134: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_413, sum_146);  mul_413 = sum_146 = None
    sub_135: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_134, mul_415);  sub_134 = mul_415 = None
    mul_416: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_135);  div_73 = sub_135 = None
    mul_417: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, mul_115);  mul_115 = None
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_246: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_243, mul_416);  add_243 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
    mul_418: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_419: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_246, mul_418);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_743: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_419, [512, 1024]);  mul_419 = None
    mm_92: "f32[512, 1024]" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
    permute_514: "f32[1024, 512]" = torch.ops.aten.permute.default(view_743, [1, 0])
    mm_93: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
    permute_515: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
    view_744: "f32[1024]" = torch.ops.aten.reshape.default(sum_150, [1024]);  sum_150 = None
    permute_516: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_745: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_92, [1, 512, 1024]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_746: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_745, [1, 512, 16, 64]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
    
    # No stacktrace found for following nodes
    view_default_90: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_517, [16, 512, 64]);  permute_517 = None
    bmm_default_44: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_43, view_default_90);  permute_default_43 = None
    view_default_91: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_44, [1, 16, 512, 64]);  bmm_default_44 = None
    bmm_default_45: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_90, permute_default_44);  view_default_90 = permute_default_44 = None
    view_default_92: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_45, [1, 16, 512, 512]);  bmm_default_45 = None
    mul_tensor_29: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_92, mul_tensor_28);  view_default_92 = mul_tensor_28 = None
    mul_tensor_30: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_29, alias_default_15);  mul_tensor_29 = None
    sum_dim_int_list_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [-1], True)
    mul_tensor_31: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_15, sum_dim_int_list_15);  alias_default_15 = sum_dim_int_list_15 = None
    sub_tensor_15: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
    view_default_93: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_15, [16, 512, 512]);  sub_tensor_15 = None
    bmm_default_46: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_45, view_default_93);  permute_default_45 = None
    view_default_94: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_46, [1, 16, 64, 512]);  bmm_default_46 = None
    mul_scalar_30: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_94, 0.3535533905932738);  view_default_94 = None
    permute_default_47: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_30, [0, 1, 3, 2]);  mul_scalar_30 = None
    bmm_default_47: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_93, permute_default_46);  view_default_93 = permute_default_46 = None
    view_default_95: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_47, [1, 16, 512, 64]);  bmm_default_47 = None
    mul_scalar_31: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_95, 0.3535533905932738);  view_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_31, [0, 2, 1, 3]);  mul_scalar_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_753: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_62, [1, 512, 1024]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_91, [0, 2, 1, 3]);  view_default_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_63: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_754: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_63, [1, 512, 1024]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_755: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_754, [512, 1024]);  view_754 = None
    mm_94: "f32[512, 1024]" = torch.ops.aten.mm.default(view_755, permute_525);  permute_525 = None
    permute_526: "f32[1024, 512]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_95: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_526, view_352);  permute_526 = None
    permute_527: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
    view_756: "f32[1024]" = torch.ops.aten.reshape.default(sum_152, [1024]);  sum_152 = None
    permute_528: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_757: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_94, [1, 512, 1024]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_47, [0, 2, 1, 3]);  permute_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_758: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_529, [1, 512, 1024]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_759: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_758, [512, 1024]);  view_758 = None
    mm_96: "f32[512, 1024]" = torch.ops.aten.mm.default(view_759, permute_530);  permute_530 = None
    permute_531: "f32[1024, 512]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_97: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_531, view_352);  permute_531 = None
    permute_532: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
    view_760: "f32[1024]" = torch.ops.aten.reshape.default(sum_153, [1024]);  sum_153 = None
    permute_533: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_761: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_96, [1, 512, 1024]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_247: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_757, view_761);  view_757 = view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_762: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_753, [512, 1024]);  view_753 = None
    mm_98: "f32[512, 1024]" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
    permute_535: "f32[1024, 512]" = torch.ops.aten.permute.default(view_762, [1, 0])
    mm_99: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
    permute_536: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_154: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
    view_763: "f32[1024]" = torch.ops.aten.reshape.default(sum_154, [1024]);  sum_154 = None
    permute_537: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_764: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_98, [1, 512, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_248: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_247, view_764);  add_247 = view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_425: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, primals_260);  primals_260 = None
    mul_426: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_425, 1024)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_425, mul_113);  mul_425 = None
    sum_156: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, sum_156);  sum_156 = None
    sub_138: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_426, sum_155);  mul_426 = sum_155 = None
    sub_139: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_138, mul_428);  sub_138 = mul_428 = None
    mul_429: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_139);  div_75 = sub_139 = None
    mul_430: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, mul_113);  mul_113 = None
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_249: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_246, mul_429);  add_246 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
    mul_431: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_432: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_249, mul_431);  mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_765: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_432, [512, 1024]);  mul_432 = None
    mm_100: "f32[512, 4096]" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
    permute_539: "f32[1024, 512]" = torch.ops.aten.permute.default(view_765, [1, 0])
    mm_101: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
    permute_540: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_159: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
    view_766: "f32[1024]" = torch.ops.aten.reshape.default(sum_159, [1024]);  sum_159 = None
    permute_541: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    view_767: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_100, [1, 512, 4096]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_434: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_435: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, view_349)
    mul_436: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
    exp_35: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_436);  mul_436 = None
    mul_437: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_438: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, mul_437);  view_349 = mul_437 = None
    add_251: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
    mul_439: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_767, add_251);  view_767 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_768: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_439, [512, 4096]);  mul_439 = None
    mm_102: "f32[512, 1024]" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
    permute_543: "f32[4096, 512]" = torch.ops.aten.permute.default(view_768, [1, 0])
    mm_103: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
    permute_544: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_160: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
    view_769: "f32[4096]" = torch.ops.aten.reshape.default(sum_160, [4096]);  sum_160 = None
    permute_545: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_544, [1, 0]);  permute_544 = None
    view_770: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_102, [1, 512, 1024]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_441: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, primals_254);  primals_254 = None
    mul_442: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_441, 1024)
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
    mul_443: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_441, mul_108);  mul_441 = None
    sum_162: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
    mul_444: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, sum_162);  sum_162 = None
    sub_141: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_442, sum_161);  mul_442 = sum_161 = None
    sub_142: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_141, mul_444);  sub_141 = mul_444 = None
    mul_445: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_76, sub_142);  div_76 = sub_142 = None
    mul_446: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, mul_108);  mul_108 = None
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_770, [0, 1]);  view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_252: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_249, mul_445);  add_249 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_447: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_448: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_252, mul_447);  mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_771: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_448, [512, 1024]);  mul_448 = None
    mm_104: "f32[512, 1024]" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
    permute_547: "f32[1024, 512]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_105: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
    permute_548: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_165: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
    view_772: "f32[1024]" = torch.ops.aten.reshape.default(sum_165, [1024]);  sum_165 = None
    permute_549: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_773: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_104, [1, 512, 1024]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_774: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_773, [1, 512, 16, 64]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_550: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
    
    # No stacktrace found for following nodes
    view_default_102: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_550, [16, 512, 64]);  permute_550 = None
    bmm_default_50: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_49, view_default_102);  permute_default_49 = None
    view_default_103: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_50, [1, 16, 512, 64]);  bmm_default_50 = None
    bmm_default_51: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_102, permute_default_50);  view_default_102 = permute_default_50 = None
    view_default_104: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_51, [1, 16, 512, 512]);  bmm_default_51 = None
    mul_tensor_33: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_104, mul_tensor_32);  view_default_104 = mul_tensor_32 = None
    mul_tensor_34: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_33, alias_default_17);  mul_tensor_33 = None
    sum_dim_int_list_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [-1], True)
    mul_tensor_35: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_17, sum_dim_int_list_17);  alias_default_17 = sum_dim_int_list_17 = None
    sub_tensor_17: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
    view_default_105: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_17, [16, 512, 512]);  sub_tensor_17 = None
    bmm_default_52: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_51, view_default_105);  permute_default_51 = None
    view_default_106: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_52, [1, 16, 64, 512]);  bmm_default_52 = None
    mul_scalar_34: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_106, 0.3535533905932738);  view_default_106 = None
    permute_default_53: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_34, [0, 1, 3, 2]);  mul_scalar_34 = None
    bmm_default_53: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_105, permute_default_52);  view_default_105 = permute_default_52 = None
    view_default_107: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_53, [1, 16, 512, 64]);  bmm_default_53 = None
    mul_scalar_35: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_107, 0.3535533905932738);  view_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_556: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_35, [0, 2, 1, 3]);  mul_scalar_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
    view_781: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_67, [1, 512, 1024]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_557: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_103, [0, 2, 1, 3]);  view_default_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_68: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    view_782: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_68, [1, 512, 1024]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_783: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_782, [512, 1024]);  view_782 = None
    mm_106: "f32[512, 1024]" = torch.ops.aten.mm.default(view_783, permute_558);  permute_558 = None
    permute_559: "f32[1024, 512]" = torch.ops.aten.permute.default(view_783, [1, 0])
    mm_107: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_559, view_330);  permute_559 = None
    permute_560: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_167: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_783, [0], True);  view_783 = None
    view_784: "f32[1024]" = torch.ops.aten.reshape.default(sum_167, [1024]);  sum_167 = None
    permute_561: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_785: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_106, [1, 512, 1024]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_562: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_53, [0, 2, 1, 3]);  permute_default_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_786: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_562, [1, 512, 1024]);  permute_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_787: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_786, [512, 1024]);  view_786 = None
    mm_108: "f32[512, 1024]" = torch.ops.aten.mm.default(view_787, permute_563);  permute_563 = None
    permute_564: "f32[1024, 512]" = torch.ops.aten.permute.default(view_787, [1, 0])
    mm_109: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_564, view_330);  permute_564 = None
    permute_565: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_168: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_787, [0], True);  view_787 = None
    view_788: "f32[1024]" = torch.ops.aten.reshape.default(sum_168, [1024]);  sum_168 = None
    permute_566: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_565, [1, 0]);  permute_565 = None
    view_789: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_108, [1, 512, 1024]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_253: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_785, view_789);  view_785 = view_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_790: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_781, [512, 1024]);  view_781 = None
    mm_110: "f32[512, 1024]" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
    permute_568: "f32[1024, 512]" = torch.ops.aten.permute.default(view_790, [1, 0])
    mm_111: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
    permute_569: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_169: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
    view_791: "f32[1024]" = torch.ops.aten.reshape.default(sum_169, [1024]);  sum_169 = None
    permute_570: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    view_792: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_110, [1, 512, 1024]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_254: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_253, view_792);  add_253 = view_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_454: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, primals_244);  primals_244 = None
    mul_455: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_454, 1024)
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_454, mul_106);  mul_454 = None
    sum_171: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_106, sum_171);  sum_171 = None
    sub_145: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_455, sum_170);  mul_455 = sum_170 = None
    sub_146: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_145, mul_457);  sub_145 = mul_457 = None
    mul_458: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_78, sub_146);  div_78 = sub_146 = None
    mul_459: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, mul_106);  mul_106 = None
    sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_255: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_252, mul_458);  add_252 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_460: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_461: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_255, mul_460);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_793: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_461, [512, 1024]);  mul_461 = None
    mm_112: "f32[512, 4096]" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
    permute_572: "f32[1024, 512]" = torch.ops.aten.permute.default(view_793, [1, 0])
    mm_113: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
    permute_573: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_174: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
    view_794: "f32[1024]" = torch.ops.aten.reshape.default(sum_174, [1024]);  sum_174 = None
    permute_574: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_795: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_112, [1, 512, 4096]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_463: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_464: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_465: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
    exp_36: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_465);  mul_465 = None
    mul_466: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_467: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, mul_466);  view_327 = mul_466 = None
    add_257: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
    mul_468: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_795, add_257);  view_795 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_796: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_468, [512, 4096]);  mul_468 = None
    mm_114: "f32[512, 1024]" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
    permute_576: "f32[4096, 512]" = torch.ops.aten.permute.default(view_796, [1, 0])
    mm_115: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
    permute_577: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_175: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
    view_797: "f32[4096]" = torch.ops.aten.reshape.default(sum_175, [4096]);  sum_175 = None
    permute_578: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_798: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_114, [1, 512, 1024]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_470: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, primals_238);  primals_238 = None
    mul_471: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_470, 1024)
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
    mul_472: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_470, mul_101);  mul_470 = None
    sum_177: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
    mul_473: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_101, sum_177);  sum_177 = None
    sub_148: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_471, sum_176);  mul_471 = sum_176 = None
    sub_149: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_148, mul_473);  sub_148 = mul_473 = None
    mul_474: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_79, sub_149);  div_79 = sub_149 = None
    mul_475: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, mul_101);  mul_101 = None
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
    sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_798, [0, 1]);  view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_258: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_255, mul_474);  add_255 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_476: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_477: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_258, mul_476);  mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_799: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_477, [512, 1024]);  mul_477 = None
    mm_116: "f32[512, 1024]" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
    permute_580: "f32[1024, 512]" = torch.ops.aten.permute.default(view_799, [1, 0])
    mm_117: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
    permute_581: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
    view_800: "f32[1024]" = torch.ops.aten.reshape.default(sum_180, [1024]);  sum_180 = None
    permute_582: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_801: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_116, [1, 512, 1024]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_802: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_801, [1, 512, 16, 64]);  view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_583: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
    
    # No stacktrace found for following nodes
    view_default_114: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_583, [16, 512, 64]);  permute_583 = None
    bmm_default_56: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_55, view_default_114);  permute_default_55 = None
    view_default_115: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_56, [1, 16, 512, 64]);  bmm_default_56 = None
    bmm_default_57: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_114, permute_default_56);  view_default_114 = permute_default_56 = None
    view_default_116: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_57, [1, 16, 512, 512]);  bmm_default_57 = None
    mul_tensor_37: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_116, mul_tensor_36);  view_default_116 = mul_tensor_36 = None
    mul_tensor_38: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, alias_default_19);  mul_tensor_37 = None
    sum_dim_int_list_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [-1], True)
    mul_tensor_39: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_19, sum_dim_int_list_19);  alias_default_19 = sum_dim_int_list_19 = None
    sub_tensor_19: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
    view_default_117: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_19, [16, 512, 512]);  sub_tensor_19 = None
    bmm_default_58: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_57, view_default_117);  permute_default_57 = None
    view_default_118: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_58, [1, 16, 64, 512]);  bmm_default_58 = None
    mul_scalar_38: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_118, 0.3535533905932738);  view_default_118 = None
    permute_default_59: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_38, [0, 1, 3, 2]);  mul_scalar_38 = None
    bmm_default_59: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_117, permute_default_58);  view_default_117 = permute_default_58 = None
    view_default_119: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_59, [1, 16, 512, 64]);  bmm_default_59 = None
    mul_scalar_39: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_119, 0.3535533905932738);  view_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_589: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_39, [0, 2, 1, 3]);  mul_scalar_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
    view_809: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_72, [1, 512, 1024]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_590: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_115, [0, 2, 1, 3]);  view_default_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_73: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_590, memory_format = torch.contiguous_format);  permute_590 = None
    view_810: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_73, [1, 512, 1024]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_811: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_810, [512, 1024]);  view_810 = None
    mm_118: "f32[512, 1024]" = torch.ops.aten.mm.default(view_811, permute_591);  permute_591 = None
    permute_592: "f32[1024, 512]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_119: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_592, view_308);  permute_592 = None
    permute_593: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[1024]" = torch.ops.aten.reshape.default(sum_182, [1024]);  sum_182 = None
    permute_594: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_593, [1, 0]);  permute_593 = None
    view_813: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_118, [1, 512, 1024]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_595: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_59, [0, 2, 1, 3]);  permute_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_814: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_595, [1, 512, 1024]);  permute_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_815: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_814, [512, 1024]);  view_814 = None
    mm_120: "f32[512, 1024]" = torch.ops.aten.mm.default(view_815, permute_596);  permute_596 = None
    permute_597: "f32[1024, 512]" = torch.ops.aten.permute.default(view_815, [1, 0])
    mm_121: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_597, view_308);  permute_597 = None
    permute_598: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_815, [0], True);  view_815 = None
    view_816: "f32[1024]" = torch.ops.aten.reshape.default(sum_183, [1024]);  sum_183 = None
    permute_599: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_817: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_120, [1, 512, 1024]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_259: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_813, view_817);  view_813 = view_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_818: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_809, [512, 1024]);  view_809 = None
    mm_122: "f32[512, 1024]" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
    permute_601: "f32[1024, 512]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_123: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
    permute_602: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_184: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[1024]" = torch.ops.aten.reshape.default(sum_184, [1024]);  sum_184 = None
    permute_603: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_820: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_122, [1, 512, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_260: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_259, view_820);  add_259 = view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_483: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, primals_228);  primals_228 = None
    mul_484: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_483, 1024)
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_483, mul_99);  mul_483 = None
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, sum_186);  sum_186 = None
    sub_152: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_484, sum_185);  mul_484 = sum_185 = None
    sub_153: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_152, mul_486);  sub_152 = mul_486 = None
    mul_487: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_81, sub_153);  div_81 = sub_153 = None
    mul_488: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, mul_99);  mul_99 = None
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_261: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_258, mul_487);  add_258 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_489: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_490: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_261, mul_489);  mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_821: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_490, [512, 1024]);  mul_490 = None
    mm_124: "f32[512, 4096]" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
    permute_605: "f32[1024, 512]" = torch.ops.aten.permute.default(view_821, [1, 0])
    mm_125: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
    permute_606: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_189: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
    view_822: "f32[1024]" = torch.ops.aten.reshape.default(sum_189, [1024]);  sum_189 = None
    permute_607: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_823: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_124, [1, 512, 4096]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_492: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
    mul_493: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, view_305)
    mul_494: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_493, -0.5);  mul_493 = None
    exp_37: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_494);  mul_494 = None
    mul_495: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_496: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, mul_495);  view_305 = mul_495 = None
    add_263: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None
    mul_497: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_823, add_263);  view_823 = add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_824: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_497, [512, 4096]);  mul_497 = None
    mm_126: "f32[512, 1024]" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
    permute_609: "f32[4096, 512]" = torch.ops.aten.permute.default(view_824, [1, 0])
    mm_127: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
    permute_610: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_190: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
    view_825: "f32[4096]" = torch.ops.aten.reshape.default(sum_190, [4096]);  sum_190 = None
    permute_611: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_610, [1, 0]);  permute_610 = None
    view_826: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_126, [1, 512, 1024]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_499: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, primals_222);  primals_222 = None
    mul_500: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_499, 1024)
    sum_191: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
    mul_501: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_499, mul_94);  mul_499 = None
    sum_192: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
    mul_502: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, sum_192);  sum_192 = None
    sub_155: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_500, sum_191);  mul_500 = sum_191 = None
    sub_156: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_155, mul_502);  sub_155 = mul_502 = None
    mul_503: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_82, sub_156);  div_82 = sub_156 = None
    mul_504: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, mul_94);  mul_94 = None
    sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_826, [0, 1]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_264: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_261, mul_503);  add_261 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_505: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_506: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_264, mul_505);  mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_827: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_506, [512, 1024]);  mul_506 = None
    mm_128: "f32[512, 1024]" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
    permute_613: "f32[1024, 512]" = torch.ops.aten.permute.default(view_827, [1, 0])
    mm_129: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
    permute_614: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_195: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
    view_828: "f32[1024]" = torch.ops.aten.reshape.default(sum_195, [1024]);  sum_195 = None
    permute_615: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_614, [1, 0]);  permute_614 = None
    view_829: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_128, [1, 512, 1024]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_830: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_829, [1, 512, 16, 64]);  view_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_616: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None
    
    # No stacktrace found for following nodes
    view_default_126: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_616, [16, 512, 64]);  permute_616 = None
    bmm_default_62: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_61, view_default_126);  permute_default_61 = None
    view_default_127: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_62, [1, 16, 512, 64]);  bmm_default_62 = None
    bmm_default_63: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_126, permute_default_62);  view_default_126 = permute_default_62 = None
    view_default_128: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_63, [1, 16, 512, 512]);  bmm_default_63 = None
    mul_tensor_41: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_128, mul_tensor_40);  view_default_128 = mul_tensor_40 = None
    mul_tensor_42: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_41, alias_default_21);  mul_tensor_41 = None
    sum_dim_int_list_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [-1], True)
    mul_tensor_43: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_21, sum_dim_int_list_21);  alias_default_21 = sum_dim_int_list_21 = None
    sub_tensor_21: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
    view_default_129: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_21, [16, 512, 512]);  sub_tensor_21 = None
    bmm_default_64: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_63, view_default_129);  permute_default_63 = None
    view_default_130: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_64, [1, 16, 64, 512]);  bmm_default_64 = None
    mul_scalar_42: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_130, 0.3535533905932738);  view_default_130 = None
    permute_default_65: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_42, [0, 1, 3, 2]);  mul_scalar_42 = None
    bmm_default_65: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_129, permute_default_64);  view_default_129 = permute_default_64 = None
    view_default_131: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_65, [1, 16, 512, 64]);  bmm_default_65 = None
    mul_scalar_43: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_131, 0.3535533905932738);  view_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_622: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_43, [0, 2, 1, 3]);  mul_scalar_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_77: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
    view_837: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 512, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_623: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_127, [0, 2, 1, 3]);  view_default_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_78: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_623, memory_format = torch.contiguous_format);  permute_623 = None
    view_838: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_78, [1, 512, 1024]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_839: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_838, [512, 1024]);  view_838 = None
    mm_130: "f32[512, 1024]" = torch.ops.aten.mm.default(view_839, permute_624);  permute_624 = None
    permute_625: "f32[1024, 512]" = torch.ops.aten.permute.default(view_839, [1, 0])
    mm_131: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_625, view_286);  permute_625 = None
    permute_626: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_839, [0], True);  view_839 = None
    view_840: "f32[1024]" = torch.ops.aten.reshape.default(sum_197, [1024]);  sum_197 = None
    permute_627: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_626, [1, 0]);  permute_626 = None
    view_841: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_628: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_65, [0, 2, 1, 3]);  permute_default_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_842: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_628, [1, 512, 1024]);  permute_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_843: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_842, [512, 1024]);  view_842 = None
    mm_132: "f32[512, 1024]" = torch.ops.aten.mm.default(view_843, permute_629);  permute_629 = None
    permute_630: "f32[1024, 512]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_133: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_630, view_286);  permute_630 = None
    permute_631: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[1024]" = torch.ops.aten.reshape.default(sum_198, [1024]);  sum_198 = None
    permute_632: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_845: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 1024]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_265: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_841, view_845);  view_841 = view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_846: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_837, [512, 1024]);  view_837 = None
    mm_134: "f32[512, 1024]" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
    permute_634: "f32[1024, 512]" = torch.ops.aten.permute.default(view_846, [1, 0])
    mm_135: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
    permute_635: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_199: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
    view_847: "f32[1024]" = torch.ops.aten.reshape.default(sum_199, [1024]);  sum_199 = None
    permute_636: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
    view_848: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 1024]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_266: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_265, view_848);  add_265 = view_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_512: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, primals_212);  primals_212 = None
    mul_513: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_512, 1024)
    sum_200: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True)
    mul_514: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_512, mul_92);  mul_512 = None
    sum_201: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True);  mul_514 = None
    mul_515: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, sum_201);  sum_201 = None
    sub_159: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_513, sum_200);  mul_513 = sum_200 = None
    sub_160: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_159, mul_515);  sub_159 = mul_515 = None
    mul_516: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_84, sub_160);  div_84 = sub_160 = None
    mul_517: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, mul_92);  mul_92 = None
    sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1]);  mul_517 = None
    sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_267: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_264, mul_516);  add_264 = mul_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_518: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_519: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_267, mul_518);  mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_849: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_519, [512, 1024]);  mul_519 = None
    mm_136: "f32[512, 4096]" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
    permute_638: "f32[1024, 512]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_137: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
    permute_639: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_204: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[1024]" = torch.ops.aten.reshape.default(sum_204, [1024]);  sum_204 = None
    permute_640: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_639, [1, 0]);  permute_639 = None
    view_851: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_136, [1, 512, 4096]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_521: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_522: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, view_283)
    mul_523: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
    exp_38: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
    mul_524: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_525: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, mul_524);  view_283 = mul_524 = None
    add_269: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
    mul_526: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_851, add_269);  view_851 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_852: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_526, [512, 4096]);  mul_526 = None
    mm_138: "f32[512, 1024]" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
    permute_642: "f32[4096, 512]" = torch.ops.aten.permute.default(view_852, [1, 0])
    mm_139: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
    permute_643: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_205: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
    view_853: "f32[4096]" = torch.ops.aten.reshape.default(sum_205, [4096]);  sum_205 = None
    permute_644: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_643, [1, 0]);  permute_643 = None
    view_854: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_138, [1, 512, 1024]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_528: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, primals_206);  primals_206 = None
    mul_529: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_528, 1024)
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2], True)
    mul_530: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_528, mul_87);  mul_528 = None
    sum_207: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
    mul_531: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, sum_207);  sum_207 = None
    sub_162: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_529, sum_206);  mul_529 = sum_206 = None
    sub_163: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_162, mul_531);  sub_162 = mul_531 = None
    mul_532: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_85, sub_163);  div_85 = sub_163 = None
    mul_533: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, mul_87);  mul_87 = None
    sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 1]);  mul_533 = None
    sum_209: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_270: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_267, mul_532);  add_267 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_534: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_535: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_270, mul_534);  mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_855: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_535, [512, 1024]);  mul_535 = None
    mm_140: "f32[512, 1024]" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
    permute_646: "f32[1024, 512]" = torch.ops.aten.permute.default(view_855, [1, 0])
    mm_141: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
    permute_647: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_210: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
    view_856: "f32[1024]" = torch.ops.aten.reshape.default(sum_210, [1024]);  sum_210 = None
    permute_648: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_647, [1, 0]);  permute_647 = None
    view_857: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_140, [1, 512, 1024]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_858: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_857, [1, 512, 16, 64]);  view_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_649: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
    
    # No stacktrace found for following nodes
    view_default_138: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_649, [16, 512, 64]);  permute_649 = None
    bmm_default_68: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_67, view_default_138);  permute_default_67 = None
    view_default_139: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_68, [1, 16, 512, 64]);  bmm_default_68 = None
    bmm_default_69: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_138, permute_default_68);  view_default_138 = permute_default_68 = None
    view_default_140: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_69, [1, 16, 512, 512]);  bmm_default_69 = None
    mul_tensor_45: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_140, mul_tensor_44);  view_default_140 = mul_tensor_44 = None
    mul_tensor_46: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_45, alias_default_23);  mul_tensor_45 = None
    sum_dim_int_list_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [-1], True)
    mul_tensor_47: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_23, sum_dim_int_list_23);  alias_default_23 = sum_dim_int_list_23 = None
    sub_tensor_23: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_46, mul_tensor_47);  mul_tensor_46 = mul_tensor_47 = None
    view_default_141: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_23, [16, 512, 512]);  sub_tensor_23 = None
    bmm_default_70: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_69, view_default_141);  permute_default_69 = None
    view_default_142: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_70, [1, 16, 64, 512]);  bmm_default_70 = None
    mul_scalar_46: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_142, 0.3535533905932738);  view_default_142 = None
    permute_default_71: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_46, [0, 1, 3, 2]);  mul_scalar_46 = None
    bmm_default_71: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_141, permute_default_70);  view_default_141 = permute_default_70 = None
    view_default_143: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_71, [1, 16, 512, 64]);  bmm_default_71 = None
    mul_scalar_47: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_143, 0.3535533905932738);  view_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_655: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_47, [0, 2, 1, 3]);  mul_scalar_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_82: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
    view_865: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_82, [1, 512, 1024]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_656: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_139, [0, 2, 1, 3]);  view_default_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_83: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_656, memory_format = torch.contiguous_format);  permute_656 = None
    view_866: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_83, [1, 512, 1024]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_867: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_866, [512, 1024]);  view_866 = None
    mm_142: "f32[512, 1024]" = torch.ops.aten.mm.default(view_867, permute_657);  permute_657 = None
    permute_658: "f32[1024, 512]" = torch.ops.aten.permute.default(view_867, [1, 0])
    mm_143: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_658, view_264);  permute_658 = None
    permute_659: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_867, [0], True);  view_867 = None
    view_868: "f32[1024]" = torch.ops.aten.reshape.default(sum_212, [1024]);  sum_212 = None
    permute_660: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_659, [1, 0]);  permute_659 = None
    view_869: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 1024]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_661: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_71, [0, 2, 1, 3]);  permute_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_870: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_661, [1, 512, 1024]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_871: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_870, [512, 1024]);  view_870 = None
    mm_144: "f32[512, 1024]" = torch.ops.aten.mm.default(view_871, permute_662);  permute_662 = None
    permute_663: "f32[1024, 512]" = torch.ops.aten.permute.default(view_871, [1, 0])
    mm_145: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_663, view_264);  permute_663 = None
    permute_664: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True);  view_871 = None
    view_872: "f32[1024]" = torch.ops.aten.reshape.default(sum_213, [1024]);  sum_213 = None
    permute_665: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_664, [1, 0]);  permute_664 = None
    view_873: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 1024]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_271: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_869, view_873);  view_869 = view_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_874: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_865, [512, 1024]);  view_865 = None
    mm_146: "f32[512, 1024]" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
    permute_667: "f32[1024, 512]" = torch.ops.aten.permute.default(view_874, [1, 0])
    mm_147: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
    permute_668: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_214: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
    view_875: "f32[1024]" = torch.ops.aten.reshape.default(sum_214, [1024]);  sum_214 = None
    permute_669: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_668, [1, 0]);  permute_668 = None
    view_876: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_146, [1, 512, 1024]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_272: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_271, view_876);  add_271 = view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_541: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, primals_196);  primals_196 = None
    mul_542: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_541, 1024)
    sum_215: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_541, [2], True)
    mul_543: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_541, mul_85);  mul_541 = None
    sum_216: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True);  mul_543 = None
    mul_544: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, sum_216);  sum_216 = None
    sub_166: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_542, sum_215);  mul_542 = sum_215 = None
    sub_167: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_166, mul_544);  sub_166 = mul_544 = None
    mul_545: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_87, sub_167);  div_87 = sub_167 = None
    mul_546: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, mul_85);  mul_85 = None
    sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 1]);  mul_546 = None
    sum_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_273: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_270, mul_545);  add_270 = mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_37: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_547: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_548: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_273, mul_547);  mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_877: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_548, [512, 1024]);  mul_548 = None
    mm_148: "f32[512, 4096]" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
    permute_671: "f32[1024, 512]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_149: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
    permute_672: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_219: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[1024]" = torch.ops.aten.reshape.default(sum_219, [1024]);  sum_219 = None
    permute_673: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
    view_879: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_148, [1, 512, 4096]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_550: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_551: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_552: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_551, -0.5);  mul_551 = None
    exp_39: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_552);  mul_552 = None
    mul_553: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_554: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, mul_553);  view_261 = mul_553 = None
    add_275: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_550, mul_554);  mul_550 = mul_554 = None
    mul_555: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_879, add_275);  view_879 = add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_880: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_555, [512, 4096]);  mul_555 = None
    mm_150: "f32[512, 1024]" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
    permute_675: "f32[4096, 512]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_151: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
    permute_676: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_220: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[4096]" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
    permute_677: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_676, [1, 0]);  permute_676 = None
    view_882: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_150, [1, 512, 1024]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_557: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, primals_190);  primals_190 = None
    mul_558: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_557, 1024)
    sum_221: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
    mul_559: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_557, mul_80);  mul_557 = None
    sum_222: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
    mul_560: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_80, sum_222);  sum_222 = None
    sub_169: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_558, sum_221);  mul_558 = sum_221 = None
    sub_170: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_169, mul_560);  sub_169 = mul_560 = None
    mul_561: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_88, sub_170);  div_88 = sub_170 = None
    mul_562: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, mul_80);  mul_80 = None
    sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
    sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_276: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_273, mul_561);  add_273 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_563: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_564: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_276, mul_563);  mul_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_883: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_564, [512, 1024]);  mul_564 = None
    mm_152: "f32[512, 1024]" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
    permute_679: "f32[1024, 512]" = torch.ops.aten.permute.default(view_883, [1, 0])
    mm_153: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
    permute_680: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_225: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
    view_884: "f32[1024]" = torch.ops.aten.reshape.default(sum_225, [1024]);  sum_225 = None
    permute_681: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_680, [1, 0]);  permute_680 = None
    view_885: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_152, [1, 512, 1024]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_886: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_885, [1, 512, 16, 64]);  view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_682: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
    
    # No stacktrace found for following nodes
    view_default_150: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_682, [16, 512, 64]);  permute_682 = None
    bmm_default_74: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_73, view_default_150);  permute_default_73 = None
    view_default_151: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_74, [1, 16, 512, 64]);  bmm_default_74 = None
    bmm_default_75: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_150, permute_default_74);  view_default_150 = permute_default_74 = None
    view_default_152: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_75, [1, 16, 512, 512]);  bmm_default_75 = None
    mul_tensor_49: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_152, mul_tensor_48);  view_default_152 = mul_tensor_48 = None
    mul_tensor_50: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_49, alias_default_25);  mul_tensor_49 = None
    sum_dim_int_list_25: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [-1], True)
    mul_tensor_51: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_25, sum_dim_int_list_25);  alias_default_25 = sum_dim_int_list_25 = None
    sub_tensor_25: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
    view_default_153: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_25, [16, 512, 512]);  sub_tensor_25 = None
    bmm_default_76: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_75, view_default_153);  permute_default_75 = None
    view_default_154: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_76, [1, 16, 64, 512]);  bmm_default_76 = None
    mul_scalar_50: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_154, 0.3535533905932738);  view_default_154 = None
    permute_default_77: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_50, [0, 1, 3, 2]);  mul_scalar_50 = None
    bmm_default_77: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_153, permute_default_76);  view_default_153 = permute_default_76 = None
    view_default_155: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_77, [1, 16, 512, 64]);  bmm_default_77 = None
    mul_scalar_51: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_155, 0.3535533905932738);  view_default_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_688: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_51, [0, 2, 1, 3]);  mul_scalar_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_87: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
    view_893: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_87, [1, 512, 1024]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_689: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_151, [0, 2, 1, 3]);  view_default_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_88: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
    view_894: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_88, [1, 512, 1024]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_895: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_894, [512, 1024]);  view_894 = None
    mm_154: "f32[512, 1024]" = torch.ops.aten.mm.default(view_895, permute_690);  permute_690 = None
    permute_691: "f32[1024, 512]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_155: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_691, view_242);  permute_691 = None
    permute_692: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_227: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[1024]" = torch.ops.aten.reshape.default(sum_227, [1024]);  sum_227 = None
    permute_693: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    view_897: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_154, [1, 512, 1024]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_694: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_77, [0, 2, 1, 3]);  permute_default_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_898: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_694, [1, 512, 1024]);  permute_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_899: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_898, [512, 1024]);  view_898 = None
    mm_156: "f32[512, 1024]" = torch.ops.aten.mm.default(view_899, permute_695);  permute_695 = None
    permute_696: "f32[1024, 512]" = torch.ops.aten.permute.default(view_899, [1, 0])
    mm_157: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_696, view_242);  permute_696 = None
    permute_697: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_899, [0], True);  view_899 = None
    view_900: "f32[1024]" = torch.ops.aten.reshape.default(sum_228, [1024]);  sum_228 = None
    permute_698: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_697, [1, 0]);  permute_697 = None
    view_901: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_156, [1, 512, 1024]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_277: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_897, view_901);  view_897 = view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_902: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_893, [512, 1024]);  view_893 = None
    mm_158: "f32[512, 1024]" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
    permute_700: "f32[1024, 512]" = torch.ops.aten.permute.default(view_902, [1, 0])
    mm_159: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
    permute_701: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_229: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
    view_903: "f32[1024]" = torch.ops.aten.reshape.default(sum_229, [1024]);  sum_229 = None
    permute_702: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_701, [1, 0]);  permute_701 = None
    view_904: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_158, [1, 512, 1024]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_278: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_277, view_904);  add_277 = view_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_570: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, primals_180);  primals_180 = None
    mul_571: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_570, 1024)
    sum_230: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True)
    mul_572: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_570, mul_78);  mul_570 = None
    sum_231: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
    mul_573: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, sum_231);  sum_231 = None
    sub_173: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_571, sum_230);  mul_571 = sum_230 = None
    sub_174: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_173, mul_573);  sub_173 = mul_573 = None
    mul_574: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_90, sub_174);  div_90 = sub_174 = None
    mul_575: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_78);  mul_78 = None
    sum_232: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
    sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_279: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_276, mul_574);  add_276 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_40: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_576: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_577: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_279, mul_576);  mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_905: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_577, [512, 1024]);  mul_577 = None
    mm_160: "f32[512, 4096]" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
    permute_704: "f32[1024, 512]" = torch.ops.aten.permute.default(view_905, [1, 0])
    mm_161: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
    permute_705: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_234: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
    view_906: "f32[1024]" = torch.ops.aten.reshape.default(sum_234, [1024]);  sum_234 = None
    permute_706: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_705, [1, 0]);  permute_705 = None
    view_907: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_160, [1, 512, 4096]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_579: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_580: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_581: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_580, -0.5);  mul_580 = None
    exp_40: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_581);  mul_581 = None
    mul_582: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_583: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, mul_582);  view_239 = mul_582 = None
    add_281: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_579, mul_583);  mul_579 = mul_583 = None
    mul_584: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_907, add_281);  view_907 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_908: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_584, [512, 4096]);  mul_584 = None
    mm_162: "f32[512, 1024]" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
    permute_708: "f32[4096, 512]" = torch.ops.aten.permute.default(view_908, [1, 0])
    mm_163: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
    permute_709: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_235: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
    view_909: "f32[4096]" = torch.ops.aten.reshape.default(sum_235, [4096]);  sum_235 = None
    permute_710: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_709, [1, 0]);  permute_709 = None
    view_910: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_162, [1, 512, 1024]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_586: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, primals_174);  primals_174 = None
    mul_587: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_586, 1024)
    sum_236: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True)
    mul_588: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_586, mul_73);  mul_586 = None
    sum_237: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True);  mul_588 = None
    mul_589: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_73, sum_237);  sum_237 = None
    sub_176: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_587, sum_236);  mul_587 = sum_236 = None
    sub_177: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_176, mul_589);  sub_176 = mul_589 = None
    mul_590: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_91, sub_177);  div_91 = sub_177 = None
    mul_591: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, mul_73);  mul_73 = None
    sum_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1]);  mul_591 = None
    sum_239: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_910, [0, 1]);  view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_282: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_279, mul_590);  add_279 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_41: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_592: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_593: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_282, mul_592);  mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_911: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_593, [512, 1024]);  mul_593 = None
    mm_164: "f32[512, 1024]" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
    permute_712: "f32[1024, 512]" = torch.ops.aten.permute.default(view_911, [1, 0])
    mm_165: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
    permute_713: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
    view_912: "f32[1024]" = torch.ops.aten.reshape.default(sum_240, [1024]);  sum_240 = None
    permute_714: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_713, [1, 0]);  permute_713 = None
    view_913: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_164, [1, 512, 1024]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_914: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_913, [1, 512, 16, 64]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_715: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None
    
    # No stacktrace found for following nodes
    view_default_162: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_715, [16, 512, 64]);  permute_715 = None
    bmm_default_80: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_79, view_default_162);  permute_default_79 = None
    view_default_163: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_80, [1, 16, 512, 64]);  bmm_default_80 = None
    bmm_default_81: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_162, permute_default_80);  view_default_162 = permute_default_80 = None
    view_default_164: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_81, [1, 16, 512, 512]);  bmm_default_81 = None
    mul_tensor_53: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_164, mul_tensor_52);  view_default_164 = mul_tensor_52 = None
    mul_tensor_54: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_53, alias_default_27);  mul_tensor_53 = None
    sum_dim_int_list_27: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [-1], True)
    mul_tensor_55: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_27, sum_dim_int_list_27);  alias_default_27 = sum_dim_int_list_27 = None
    sub_tensor_27: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_54, mul_tensor_55);  mul_tensor_54 = mul_tensor_55 = None
    view_default_165: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_27, [16, 512, 512]);  sub_tensor_27 = None
    bmm_default_82: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_81, view_default_165);  permute_default_81 = None
    view_default_166: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_82, [1, 16, 64, 512]);  bmm_default_82 = None
    mul_scalar_54: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_166, 0.3535533905932738);  view_default_166 = None
    permute_default_83: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_54, [0, 1, 3, 2]);  mul_scalar_54 = None
    bmm_default_83: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_165, permute_default_82);  view_default_165 = permute_default_82 = None
    view_default_167: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_83, [1, 16, 512, 64]);  bmm_default_83 = None
    mul_scalar_55: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_167, 0.3535533905932738);  view_default_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_721: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_55, [0, 2, 1, 3]);  mul_scalar_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_92: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
    view_921: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_92, [1, 512, 1024]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_722: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_163, [0, 2, 1, 3]);  view_default_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_93: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_722, memory_format = torch.contiguous_format);  permute_722 = None
    view_922: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_93, [1, 512, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_923: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_922, [512, 1024]);  view_922 = None
    mm_166: "f32[512, 1024]" = torch.ops.aten.mm.default(view_923, permute_723);  permute_723 = None
    permute_724: "f32[1024, 512]" = torch.ops.aten.permute.default(view_923, [1, 0])
    mm_167: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_724, view_220);  permute_724 = None
    permute_725: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_923, [0], True);  view_923 = None
    view_924: "f32[1024]" = torch.ops.aten.reshape.default(sum_242, [1024]);  sum_242 = None
    permute_726: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_725, [1, 0]);  permute_725 = None
    view_925: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_166, [1, 512, 1024]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_727: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_83, [0, 2, 1, 3]);  permute_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_926: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_727, [1, 512, 1024]);  permute_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_927: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_926, [512, 1024]);  view_926 = None
    mm_168: "f32[512, 1024]" = torch.ops.aten.mm.default(view_927, permute_728);  permute_728 = None
    permute_729: "f32[1024, 512]" = torch.ops.aten.permute.default(view_927, [1, 0])
    mm_169: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_729, view_220);  permute_729 = None
    permute_730: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_927, [0], True);  view_927 = None
    view_928: "f32[1024]" = torch.ops.aten.reshape.default(sum_243, [1024]);  sum_243 = None
    permute_731: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_730, [1, 0]);  permute_730 = None
    view_929: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_168, [1, 512, 1024]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_283: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_925, view_929);  view_925 = view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_930: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_921, [512, 1024]);  view_921 = None
    mm_170: "f32[512, 1024]" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
    permute_733: "f32[1024, 512]" = torch.ops.aten.permute.default(view_930, [1, 0])
    mm_171: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
    permute_734: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_244: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
    view_931: "f32[1024]" = torch.ops.aten.reshape.default(sum_244, [1024]);  sum_244 = None
    permute_735: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
    view_932: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_170, [1, 512, 1024]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_284: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_283, view_932);  add_283 = view_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_599: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, primals_164);  primals_164 = None
    mul_600: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_599, 1024)
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True)
    mul_601: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_599, mul_71);  mul_599 = None
    sum_246: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True);  mul_601 = None
    mul_602: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_71, sum_246);  sum_246 = None
    sub_180: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_600, sum_245);  mul_600 = sum_245 = None
    sub_181: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_180, mul_602);  sub_180 = mul_602 = None
    mul_603: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_93, sub_181);  div_93 = sub_181 = None
    mul_604: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, mul_71);  mul_71 = None
    sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 1]);  mul_604 = None
    sum_248: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_285: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_282, mul_603);  add_282 = mul_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_43: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_605: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_606: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_285, mul_605);  mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_933: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_606, [512, 1024]);  mul_606 = None
    mm_172: "f32[512, 4096]" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
    permute_737: "f32[1024, 512]" = torch.ops.aten.permute.default(view_933, [1, 0])
    mm_173: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
    permute_738: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_249: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
    view_934: "f32[1024]" = torch.ops.aten.reshape.default(sum_249, [1024]);  sum_249 = None
    permute_739: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_738, [1, 0]);  permute_738 = None
    view_935: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_172, [1, 512, 4096]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_608: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_609: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_610: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_609, -0.5);  mul_609 = None
    exp_41: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_610);  mul_610 = None
    mul_611: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_612: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, mul_611);  view_217 = mul_611 = None
    add_287: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_608, mul_612);  mul_608 = mul_612 = None
    mul_613: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_935, add_287);  view_935 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_936: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_613, [512, 4096]);  mul_613 = None
    mm_174: "f32[512, 1024]" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
    permute_741: "f32[4096, 512]" = torch.ops.aten.permute.default(view_936, [1, 0])
    mm_175: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
    permute_742: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_250: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
    view_937: "f32[4096]" = torch.ops.aten.reshape.default(sum_250, [4096]);  sum_250 = None
    permute_743: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_742, [1, 0]);  permute_742 = None
    view_938: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_174, [1, 512, 1024]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_615: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, primals_158);  primals_158 = None
    mul_616: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_615, 1024)
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_615, [2], True)
    mul_617: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_615, mul_66);  mul_615 = None
    sum_252: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True);  mul_617 = None
    mul_618: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, sum_252);  sum_252 = None
    sub_183: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_616, sum_251);  mul_616 = sum_251 = None
    sub_184: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_183, mul_618);  sub_183 = mul_618 = None
    mul_619: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_94, sub_184);  div_94 = sub_184 = None
    mul_620: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, mul_66);  mul_66 = None
    sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1]);  mul_620 = None
    sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_938, [0, 1]);  view_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_288: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_285, mul_619);  add_285 = mul_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_44: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_621: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_622: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_288, mul_621);  mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_939: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_622, [512, 1024]);  mul_622 = None
    mm_176: "f32[512, 1024]" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
    permute_745: "f32[1024, 512]" = torch.ops.aten.permute.default(view_939, [1, 0])
    mm_177: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
    permute_746: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
    view_940: "f32[1024]" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
    permute_747: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_746, [1, 0]);  permute_746 = None
    view_941: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_176, [1, 512, 1024]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_942: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_941, [1, 512, 16, 64]);  view_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_748: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None
    
    # No stacktrace found for following nodes
    view_default_174: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_748, [16, 512, 64]);  permute_748 = None
    bmm_default_86: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_85, view_default_174);  permute_default_85 = None
    view_default_175: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_86, [1, 16, 512, 64]);  bmm_default_86 = None
    bmm_default_87: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_174, permute_default_86);  view_default_174 = permute_default_86 = None
    view_default_176: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_87, [1, 16, 512, 512]);  bmm_default_87 = None
    mul_tensor_57: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_176, mul_tensor_56);  view_default_176 = mul_tensor_56 = None
    mul_tensor_58: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_57, alias_default_29);  mul_tensor_57 = None
    sum_dim_int_list_29: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_58, [-1], True)
    mul_tensor_59: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_29, sum_dim_int_list_29);  alias_default_29 = sum_dim_int_list_29 = None
    sub_tensor_29: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
    view_default_177: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_29, [16, 512, 512]);  sub_tensor_29 = None
    bmm_default_88: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_87, view_default_177);  permute_default_87 = None
    view_default_178: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_88, [1, 16, 64, 512]);  bmm_default_88 = None
    mul_scalar_58: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_178, 0.3535533905932738);  view_default_178 = None
    permute_default_89: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_58, [0, 1, 3, 2]);  mul_scalar_58 = None
    bmm_default_89: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_177, permute_default_88);  view_default_177 = permute_default_88 = None
    view_default_179: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_89, [1, 16, 512, 64]);  bmm_default_89 = None
    mul_scalar_59: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_179, 0.3535533905932738);  view_default_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_754: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_59, [0, 2, 1, 3]);  mul_scalar_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_97: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
    view_949: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_97, [1, 512, 1024]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_755: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_175, [0, 2, 1, 3]);  view_default_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_98: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
    view_950: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_98, [1, 512, 1024]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_951: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_950, [512, 1024]);  view_950 = None
    mm_178: "f32[512, 1024]" = torch.ops.aten.mm.default(view_951, permute_756);  permute_756 = None
    permute_757: "f32[1024, 512]" = torch.ops.aten.permute.default(view_951, [1, 0])
    mm_179: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_757, view_198);  permute_757 = None
    permute_758: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_257: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_951, [0], True);  view_951 = None
    view_952: "f32[1024]" = torch.ops.aten.reshape.default(sum_257, [1024]);  sum_257 = None
    permute_759: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    view_953: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_178, [1, 512, 1024]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_760: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_89, [0, 2, 1, 3]);  permute_default_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_954: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_760, [1, 512, 1024]);  permute_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_955: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_954, [512, 1024]);  view_954 = None
    mm_180: "f32[512, 1024]" = torch.ops.aten.mm.default(view_955, permute_761);  permute_761 = None
    permute_762: "f32[1024, 512]" = torch.ops.aten.permute.default(view_955, [1, 0])
    mm_181: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_762, view_198);  permute_762 = None
    permute_763: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_955, [0], True);  view_955 = None
    view_956: "f32[1024]" = torch.ops.aten.reshape.default(sum_258, [1024]);  sum_258 = None
    permute_764: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_763, [1, 0]);  permute_763 = None
    view_957: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_180, [1, 512, 1024]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_289: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_953, view_957);  view_953 = view_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_958: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_949, [512, 1024]);  view_949 = None
    mm_182: "f32[512, 1024]" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
    permute_766: "f32[1024, 512]" = torch.ops.aten.permute.default(view_958, [1, 0])
    mm_183: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
    permute_767: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_259: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
    view_959: "f32[1024]" = torch.ops.aten.reshape.default(sum_259, [1024]);  sum_259 = None
    permute_768: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_767, [1, 0]);  permute_767 = None
    view_960: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_182, [1, 512, 1024]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_290: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_289, view_960);  add_289 = view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_628: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_148);  primals_148 = None
    mul_629: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_628, 1024)
    sum_260: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True)
    mul_630: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_628, mul_64);  mul_628 = None
    sum_261: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True);  mul_630 = None
    mul_631: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_64, sum_261);  sum_261 = None
    sub_187: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_629, sum_260);  mul_629 = sum_260 = None
    sub_188: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_187, mul_631);  sub_187 = mul_631 = None
    mul_632: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_96, sub_188);  div_96 = sub_188 = None
    mul_633: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_64);  mul_64 = None
    sum_262: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 1]);  mul_633 = None
    sum_263: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_291: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_288, mul_632);  add_288 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_46: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_634: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_635: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_291, mul_634);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_961: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_635, [512, 1024]);  mul_635 = None
    mm_184: "f32[512, 4096]" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
    permute_770: "f32[1024, 512]" = torch.ops.aten.permute.default(view_961, [1, 0])
    mm_185: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
    permute_771: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_264: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
    view_962: "f32[1024]" = torch.ops.aten.reshape.default(sum_264, [1024]);  sum_264 = None
    permute_772: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_771, [1, 0]);  permute_771 = None
    view_963: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_184, [1, 512, 4096]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_637: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_638: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_639: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_638, -0.5);  mul_638 = None
    exp_42: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_639);  mul_639 = None
    mul_640: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_641: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, mul_640);  view_195 = mul_640 = None
    add_293: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_637, mul_641);  mul_637 = mul_641 = None
    mul_642: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_963, add_293);  view_963 = add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_964: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_642, [512, 4096]);  mul_642 = None
    mm_186: "f32[512, 1024]" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
    permute_774: "f32[4096, 512]" = torch.ops.aten.permute.default(view_964, [1, 0])
    mm_187: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
    permute_775: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_265: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
    view_965: "f32[4096]" = torch.ops.aten.reshape.default(sum_265, [4096]);  sum_265 = None
    permute_776: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_775, [1, 0]);  permute_775 = None
    view_966: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_186, [1, 512, 1024]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_644: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, primals_142);  primals_142 = None
    mul_645: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_644, 1024)
    sum_266: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
    mul_646: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_644, mul_59);  mul_644 = None
    sum_267: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
    mul_647: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, sum_267);  sum_267 = None
    sub_190: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_645, sum_266);  mul_645 = sum_266 = None
    sub_191: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_190, mul_647);  sub_190 = mul_647 = None
    mul_648: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_97, sub_191);  div_97 = sub_191 = None
    mul_649: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, mul_59);  mul_59 = None
    sum_268: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
    sum_269: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_294: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_291, mul_648);  add_291 = mul_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_47: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_650: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_651: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_294, mul_650);  mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_967: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_651, [512, 1024]);  mul_651 = None
    mm_188: "f32[512, 1024]" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
    permute_778: "f32[1024, 512]" = torch.ops.aten.permute.default(view_967, [1, 0])
    mm_189: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
    permute_779: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_270: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
    view_968: "f32[1024]" = torch.ops.aten.reshape.default(sum_270, [1024]);  sum_270 = None
    permute_780: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_779, [1, 0]);  permute_779 = None
    view_969: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_188, [1, 512, 1024]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_970: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_969, [1, 512, 16, 64]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_781: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
    
    # No stacktrace found for following nodes
    view_default_186: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_781, [16, 512, 64]);  permute_781 = None
    bmm_default_92: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_91, view_default_186);  permute_default_91 = None
    view_default_187: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_92, [1, 16, 512, 64]);  bmm_default_92 = None
    bmm_default_93: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_186, permute_default_92);  view_default_186 = permute_default_92 = None
    view_default_188: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_93, [1, 16, 512, 512]);  bmm_default_93 = None
    mul_tensor_61: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_188, mul_tensor_60);  view_default_188 = mul_tensor_60 = None
    mul_tensor_62: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_61, alias_default_31);  mul_tensor_61 = None
    sum_dim_int_list_31: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_62, [-1], True)
    mul_tensor_63: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_31, sum_dim_int_list_31);  alias_default_31 = sum_dim_int_list_31 = None
    sub_tensor_31: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_62, mul_tensor_63);  mul_tensor_62 = mul_tensor_63 = None
    view_default_189: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_31, [16, 512, 512]);  sub_tensor_31 = None
    bmm_default_94: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_93, view_default_189);  permute_default_93 = None
    view_default_190: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_94, [1, 16, 64, 512]);  bmm_default_94 = None
    mul_scalar_62: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_190, 0.3535533905932738);  view_default_190 = None
    permute_default_95: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_62, [0, 1, 3, 2]);  mul_scalar_62 = None
    bmm_default_95: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_189, permute_default_94);  view_default_189 = permute_default_94 = None
    view_default_191: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_95, [1, 16, 512, 64]);  bmm_default_95 = None
    mul_scalar_63: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_191, 0.3535533905932738);  view_default_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_787: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_63, [0, 2, 1, 3]);  mul_scalar_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_102: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
    view_977: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_102, [1, 512, 1024]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_788: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_187, [0, 2, 1, 3]);  view_default_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_103: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_788, memory_format = torch.contiguous_format);  permute_788 = None
    view_978: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_103, [1, 512, 1024]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_979: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_978, [512, 1024]);  view_978 = None
    mm_190: "f32[512, 1024]" = torch.ops.aten.mm.default(view_979, permute_789);  permute_789 = None
    permute_790: "f32[1024, 512]" = torch.ops.aten.permute.default(view_979, [1, 0])
    mm_191: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_790, view_176);  permute_790 = None
    permute_791: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_272: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
    view_980: "f32[1024]" = torch.ops.aten.reshape.default(sum_272, [1024]);  sum_272 = None
    permute_792: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_791, [1, 0]);  permute_791 = None
    view_981: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [1, 512, 1024]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_793: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_95, [0, 2, 1, 3]);  permute_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_982: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_793, [1, 512, 1024]);  permute_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_983: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_982, [512, 1024]);  view_982 = None
    mm_192: "f32[512, 1024]" = torch.ops.aten.mm.default(view_983, permute_794);  permute_794 = None
    permute_795: "f32[1024, 512]" = torch.ops.aten.permute.default(view_983, [1, 0])
    mm_193: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_795, view_176);  permute_795 = None
    permute_796: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_273: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_983, [0], True);  view_983 = None
    view_984: "f32[1024]" = torch.ops.aten.reshape.default(sum_273, [1024]);  sum_273 = None
    permute_797: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_796, [1, 0]);  permute_796 = None
    view_985: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_192, [1, 512, 1024]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_295: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_981, view_985);  view_981 = view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_986: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_977, [512, 1024]);  view_977 = None
    mm_194: "f32[512, 1024]" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
    permute_799: "f32[1024, 512]" = torch.ops.aten.permute.default(view_986, [1, 0])
    mm_195: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
    permute_800: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_274: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[1024]" = torch.ops.aten.reshape.default(sum_274, [1024]);  sum_274 = None
    permute_801: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
    view_988: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_194, [1, 512, 1024]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_296: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_295, view_988);  add_295 = view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_657: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, primals_132);  primals_132 = None
    mul_658: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_657, 1024)
    sum_275: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True)
    mul_659: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_657, mul_57);  mul_657 = None
    sum_276: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True);  mul_659 = None
    mul_660: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, sum_276);  sum_276 = None
    sub_194: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_658, sum_275);  mul_658 = sum_275 = None
    sub_195: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_194, mul_660);  sub_194 = mul_660 = None
    mul_661: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_99, sub_195);  div_99 = sub_195 = None
    mul_662: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, mul_57);  mul_57 = None
    sum_277: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 1]);  mul_662 = None
    sum_278: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_297: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_294, mul_661);  add_294 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_49: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_663: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_664: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_297, mul_663);  mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_989: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_664, [512, 1024]);  mul_664 = None
    mm_196: "f32[512, 4096]" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
    permute_803: "f32[1024, 512]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_197: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
    permute_804: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_279: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[1024]" = torch.ops.aten.reshape.default(sum_279, [1024]);  sum_279 = None
    permute_805: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_804, [1, 0]);  permute_804 = None
    view_991: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_196, [1, 512, 4096]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_666: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_667: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_668: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_667, -0.5);  mul_667 = None
    exp_43: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_668);  mul_668 = None
    mul_669: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_670: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, mul_669);  view_173 = mul_669 = None
    add_299: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_666, mul_670);  mul_666 = mul_670 = None
    mul_671: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_991, add_299);  view_991 = add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_992: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_671, [512, 4096]);  mul_671 = None
    mm_198: "f32[512, 1024]" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
    permute_807: "f32[4096, 512]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_199: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
    permute_808: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_280: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[4096]" = torch.ops.aten.reshape.default(sum_280, [4096]);  sum_280 = None
    permute_809: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_808, [1, 0]);  permute_808 = None
    view_994: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_198, [1, 512, 1024]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_673: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, primals_126);  primals_126 = None
    mul_674: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_673, 1024)
    sum_281: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True)
    mul_675: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_673, mul_52);  mul_673 = None
    sum_282: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True);  mul_675 = None
    mul_676: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_52, sum_282);  sum_282 = None
    sub_197: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_674, sum_281);  mul_674 = sum_281 = None
    sub_198: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_197, mul_676);  sub_197 = mul_676 = None
    mul_677: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_100, sub_198);  div_100 = sub_198 = None
    mul_678: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, mul_52);  mul_52 = None
    sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 1]);  mul_678 = None
    sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_300: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_297, mul_677);  add_297 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_50: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_679: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_680: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_300, mul_679);  mul_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_995: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_680, [512, 1024]);  mul_680 = None
    mm_200: "f32[512, 1024]" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
    permute_811: "f32[1024, 512]" = torch.ops.aten.permute.default(view_995, [1, 0])
    mm_201: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
    permute_812: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_285: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
    view_996: "f32[1024]" = torch.ops.aten.reshape.default(sum_285, [1024]);  sum_285 = None
    permute_813: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_812, [1, 0]);  permute_812 = None
    view_997: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_200, [1, 512, 1024]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_998: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_997, [1, 512, 16, 64]);  view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_814: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None
    
    # No stacktrace found for following nodes
    view_default_198: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_814, [16, 512, 64]);  permute_814 = None
    bmm_default_98: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_97, view_default_198);  permute_default_97 = None
    view_default_199: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_98, [1, 16, 512, 64]);  bmm_default_98 = None
    bmm_default_99: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_198, permute_default_98);  view_default_198 = permute_default_98 = None
    view_default_200: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_99, [1, 16, 512, 512]);  bmm_default_99 = None
    mul_tensor_65: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_200, mul_tensor_64);  view_default_200 = mul_tensor_64 = None
    mul_tensor_66: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_65, alias_default_33);  mul_tensor_65 = None
    sum_dim_int_list_33: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [-1], True)
    mul_tensor_67: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_33, sum_dim_int_list_33);  alias_default_33 = sum_dim_int_list_33 = None
    sub_tensor_33: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_66, mul_tensor_67);  mul_tensor_66 = mul_tensor_67 = None
    view_default_201: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_33, [16, 512, 512]);  sub_tensor_33 = None
    bmm_default_100: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_99, view_default_201);  permute_default_99 = None
    view_default_202: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_100, [1, 16, 64, 512]);  bmm_default_100 = None
    mul_scalar_66: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_202, 0.3535533905932738);  view_default_202 = None
    permute_default_101: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_66, [0, 1, 3, 2]);  mul_scalar_66 = None
    bmm_default_101: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_201, permute_default_100);  view_default_201 = permute_default_100 = None
    view_default_203: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_101, [1, 16, 512, 64]);  bmm_default_101 = None
    mul_scalar_67: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_203, 0.3535533905932738);  view_default_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_820: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_67, [0, 2, 1, 3]);  mul_scalar_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_107: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
    view_1005: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_107, [1, 512, 1024]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_821: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_199, [0, 2, 1, 3]);  view_default_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_108: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
    view_1006: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_108, [1, 512, 1024]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1007: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1006, [512, 1024]);  view_1006 = None
    mm_202: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1007, permute_822);  permute_822 = None
    permute_823: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_203: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_823, view_154);  permute_823 = None
    permute_824: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_287: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
    view_1008: "f32[1024]" = torch.ops.aten.reshape.default(sum_287, [1024]);  sum_287 = None
    permute_825: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_1009: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_202, [1, 512, 1024]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_826: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_101, [0, 2, 1, 3]);  permute_default_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1010: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_826, [1, 512, 1024]);  permute_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1011: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1010, [512, 1024]);  view_1010 = None
    mm_204: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1011, permute_827);  permute_827 = None
    permute_828: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1011, [1, 0])
    mm_205: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_828, view_154);  permute_828 = None
    permute_829: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_288: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1011, [0], True);  view_1011 = None
    view_1012: "f32[1024]" = torch.ops.aten.reshape.default(sum_288, [1024]);  sum_288 = None
    permute_830: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
    view_1013: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_204, [1, 512, 1024]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_301: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1009, view_1013);  view_1009 = view_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1014: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1005, [512, 1024]);  view_1005 = None
    mm_206: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
    permute_832: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_207: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
    permute_833: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_289: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[1024]" = torch.ops.aten.reshape.default(sum_289, [1024]);  sum_289 = None
    permute_834: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_833, [1, 0]);  permute_833 = None
    view_1016: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_206, [1, 512, 1024]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_302: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_301, view_1016);  add_301 = view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_686: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, primals_116);  primals_116 = None
    mul_687: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, 1024)
    sum_290: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
    mul_688: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, mul_50);  mul_686 = None
    sum_291: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
    mul_689: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_50, sum_291);  sum_291 = None
    sub_201: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_687, sum_290);  mul_687 = sum_290 = None
    sub_202: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_201, mul_689);  sub_201 = mul_689 = None
    mul_690: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_102, sub_202);  div_102 = sub_202 = None
    mul_691: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_50);  mul_50 = None
    sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
    sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_303: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_300, mul_690);  add_300 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_52: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_692: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_693: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_303, mul_692);  mul_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1017: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_693, [512, 1024]);  mul_693 = None
    mm_208: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
    permute_836: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1017, [1, 0])
    mm_209: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
    permute_837: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
    view_1018: "f32[1024]" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
    permute_838: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_837, [1, 0]);  permute_837 = None
    view_1019: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_208, [1, 512, 4096]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_695: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_696: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_697: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_696, -0.5);  mul_696 = None
    exp_44: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_697);  mul_697 = None
    mul_698: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_699: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, mul_698);  view_151 = mul_698 = None
    add_305: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_695, mul_699);  mul_695 = mul_699 = None
    mul_700: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1019, add_305);  view_1019 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1020: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_700, [512, 4096]);  mul_700 = None
    mm_210: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
    permute_840: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1020, [1, 0])
    mm_211: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
    permute_841: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_295: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
    view_1021: "f32[4096]" = torch.ops.aten.reshape.default(sum_295, [4096]);  sum_295 = None
    permute_842: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    view_1022: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_210, [1, 512, 1024]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_702: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, primals_110);  primals_110 = None
    mul_703: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, 1024)
    sum_296: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True)
    mul_704: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, mul_45);  mul_702 = None
    sum_297: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [2], True);  mul_704 = None
    mul_705: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, sum_297);  sum_297 = None
    sub_204: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_703, sum_296);  mul_703 = sum_296 = None
    sub_205: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_204, mul_705);  sub_204 = mul_705 = None
    mul_706: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_103, sub_205);  div_103 = sub_205 = None
    mul_707: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, mul_45);  mul_45 = None
    sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1]);  mul_707 = None
    sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1022, [0, 1]);  view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_306: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_303, mul_706);  add_303 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_53: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_708: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_709: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_306, mul_708);  mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1023: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_709, [512, 1024]);  mul_709 = None
    mm_212: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
    permute_844: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1023, [1, 0])
    mm_213: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
    permute_845: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_300: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
    view_1024: "f32[1024]" = torch.ops.aten.reshape.default(sum_300, [1024]);  sum_300 = None
    permute_846: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_845, [1, 0]);  permute_845 = None
    view_1025: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_212, [1, 512, 1024]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1026: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1025, [1, 512, 16, 64]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_847: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None
    
    # No stacktrace found for following nodes
    view_default_210: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_847, [16, 512, 64]);  permute_847 = None
    bmm_default_104: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_103, view_default_210);  permute_default_103 = None
    view_default_211: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_104, [1, 16, 512, 64]);  bmm_default_104 = None
    bmm_default_105: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_210, permute_default_104);  view_default_210 = permute_default_104 = None
    view_default_212: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_105, [1, 16, 512, 512]);  bmm_default_105 = None
    mul_tensor_69: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_212, mul_tensor_68);  view_default_212 = mul_tensor_68 = None
    mul_tensor_70: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_69, alias_default_35);  mul_tensor_69 = None
    sum_dim_int_list_35: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_70, [-1], True)
    mul_tensor_71: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_35, sum_dim_int_list_35);  alias_default_35 = sum_dim_int_list_35 = None
    sub_tensor_35: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
    view_default_213: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_35, [16, 512, 512]);  sub_tensor_35 = None
    bmm_default_106: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_105, view_default_213);  permute_default_105 = None
    view_default_214: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_106, [1, 16, 64, 512]);  bmm_default_106 = None
    mul_scalar_70: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_214, 0.3535533905932738);  view_default_214 = None
    permute_default_107: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_70, [0, 1, 3, 2]);  mul_scalar_70 = None
    bmm_default_107: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_213, permute_default_106);  view_default_213 = permute_default_106 = None
    view_default_215: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_107, [1, 16, 512, 64]);  bmm_default_107 = None
    mul_scalar_71: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_215, 0.3535533905932738);  view_default_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_853: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_71, [0, 2, 1, 3]);  mul_scalar_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_112: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_853, memory_format = torch.contiguous_format);  permute_853 = None
    view_1033: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_112, [1, 512, 1024]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_854: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_211, [0, 2, 1, 3]);  view_default_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_113: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_854, memory_format = torch.contiguous_format);  permute_854 = None
    view_1034: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_113, [1, 512, 1024]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1035: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1034, [512, 1024]);  view_1034 = None
    mm_214: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1035, permute_855);  permute_855 = None
    permute_856: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1035, [1, 0])
    mm_215: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_856, view_132);  permute_856 = None
    permute_857: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1035, [0], True);  view_1035 = None
    view_1036: "f32[1024]" = torch.ops.aten.reshape.default(sum_302, [1024]);  sum_302 = None
    permute_858: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_857, [1, 0]);  permute_857 = None
    view_1037: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_214, [1, 512, 1024]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_859: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_107, [0, 2, 1, 3]);  permute_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1038: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_859, [1, 512, 1024]);  permute_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1039: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1038, [512, 1024]);  view_1038 = None
    mm_216: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1039, permute_860);  permute_860 = None
    permute_861: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1039, [1, 0])
    mm_217: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_861, view_132);  permute_861 = None
    permute_862: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1039, [0], True);  view_1039 = None
    view_1040: "f32[1024]" = torch.ops.aten.reshape.default(sum_303, [1024]);  sum_303 = None
    permute_863: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_862, [1, 0]);  permute_862 = None
    view_1041: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_216, [1, 512, 1024]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_307: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1037, view_1041);  view_1037 = view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1042: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1033, [512, 1024]);  view_1033 = None
    mm_218: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
    permute_865: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1042, [1, 0])
    mm_219: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
    permute_866: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_304: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
    view_1043: "f32[1024]" = torch.ops.aten.reshape.default(sum_304, [1024]);  sum_304 = None
    permute_867: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
    view_1044: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_218, [1, 512, 1024]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_308: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_307, view_1044);  add_307 = view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_715: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, primals_100);  primals_100 = None
    mul_716: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_715, 1024)
    sum_305: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True)
    mul_717: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_715, mul_43);  mul_715 = None
    sum_306: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True);  mul_717 = None
    mul_718: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_43, sum_306);  sum_306 = None
    sub_208: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_716, sum_305);  mul_716 = sum_305 = None
    sub_209: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_208, mul_718);  sub_208 = mul_718 = None
    mul_719: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_105, sub_209);  div_105 = sub_209 = None
    mul_720: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_43);  mul_43 = None
    sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 1]);  mul_720 = None
    sum_308: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_309: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_306, mul_719);  add_306 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_55: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_721: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_722: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_309, mul_721);  mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1045: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_722, [512, 1024]);  mul_722 = None
    mm_220: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
    permute_869: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1045, [1, 0])
    mm_221: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
    permute_870: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
    view_1046: "f32[1024]" = torch.ops.aten.reshape.default(sum_309, [1024]);  sum_309 = None
    permute_871: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_870, [1, 0]);  permute_870 = None
    view_1047: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_220, [1, 512, 4096]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_724: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_725: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_726: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_725, -0.5);  mul_725 = None
    exp_45: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_726);  mul_726 = None
    mul_727: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_728: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, mul_727);  view_129 = mul_727 = None
    add_311: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_724, mul_728);  mul_724 = mul_728 = None
    mul_729: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1047, add_311);  view_1047 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1048: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_729, [512, 4096]);  mul_729 = None
    mm_222: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
    permute_873: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1048, [1, 0])
    mm_223: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
    permute_874: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_310: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
    view_1049: "f32[4096]" = torch.ops.aten.reshape.default(sum_310, [4096]);  sum_310 = None
    permute_875: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_874, [1, 0]);  permute_874 = None
    view_1050: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_222, [1, 512, 1024]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_731: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, primals_94);  primals_94 = None
    mul_732: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, 1024)
    sum_311: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
    mul_733: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, mul_38);  mul_731 = None
    sum_312: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
    mul_734: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_38, sum_312);  sum_312 = None
    sub_211: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_732, sum_311);  mul_732 = sum_311 = None
    sub_212: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_211, mul_734);  sub_211 = mul_734 = None
    mul_735: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_106, sub_212);  div_106 = sub_212 = None
    mul_736: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, mul_38);  mul_38 = None
    sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
    sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1050, [0, 1]);  view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_312: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_309, mul_735);  add_309 = mul_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_56: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_737: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_738: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_312, mul_737);  mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1051: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_738, [512, 1024]);  mul_738 = None
    mm_224: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
    permute_877: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1051, [1, 0])
    mm_225: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
    permute_878: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_315: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
    view_1052: "f32[1024]" = torch.ops.aten.reshape.default(sum_315, [1024]);  sum_315 = None
    permute_879: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_878, [1, 0]);  permute_878 = None
    view_1053: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_224, [1, 512, 1024]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1054: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1053, [1, 512, 16, 64]);  view_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_880: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None
    
    # No stacktrace found for following nodes
    view_default_222: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_880, [16, 512, 64]);  permute_880 = None
    bmm_default_110: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_109, view_default_222);  permute_default_109 = None
    view_default_223: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_110, [1, 16, 512, 64]);  bmm_default_110 = None
    bmm_default_111: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_222, permute_default_110);  view_default_222 = permute_default_110 = None
    view_default_224: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_111, [1, 16, 512, 512]);  bmm_default_111 = None
    mul_tensor_73: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_224, mul_tensor_72);  view_default_224 = mul_tensor_72 = None
    mul_tensor_74: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_73, alias_default_37);  mul_tensor_73 = None
    sum_dim_int_list_37: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_74, [-1], True)
    mul_tensor_75: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_37, sum_dim_int_list_37);  alias_default_37 = sum_dim_int_list_37 = None
    sub_tensor_37: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_74, mul_tensor_75);  mul_tensor_74 = mul_tensor_75 = None
    view_default_225: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_37, [16, 512, 512]);  sub_tensor_37 = None
    bmm_default_112: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_111, view_default_225);  permute_default_111 = None
    view_default_226: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_112, [1, 16, 64, 512]);  bmm_default_112 = None
    mul_scalar_74: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_226, 0.3535533905932738);  view_default_226 = None
    permute_default_113: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_74, [0, 1, 3, 2]);  mul_scalar_74 = None
    bmm_default_113: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_225, permute_default_112);  view_default_225 = permute_default_112 = None
    view_default_227: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_113, [1, 16, 512, 64]);  bmm_default_113 = None
    mul_scalar_75: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_227, 0.3535533905932738);  view_default_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_886: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_75, [0, 2, 1, 3]);  mul_scalar_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_117: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
    view_1061: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_117, [1, 512, 1024]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_887: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_223, [0, 2, 1, 3]);  view_default_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_118: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
    view_1062: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_118, [1, 512, 1024]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1063: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1062, [512, 1024]);  view_1062 = None
    mm_226: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1063, permute_888);  permute_888 = None
    permute_889: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1063, [1, 0])
    mm_227: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_889, view_110);  permute_889 = None
    permute_890: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1063, [0], True);  view_1063 = None
    view_1064: "f32[1024]" = torch.ops.aten.reshape.default(sum_317, [1024]);  sum_317 = None
    permute_891: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_890, [1, 0]);  permute_890 = None
    view_1065: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_226, [1, 512, 1024]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_892: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_113, [0, 2, 1, 3]);  permute_default_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1066: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_892, [1, 512, 1024]);  permute_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1067: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1066, [512, 1024]);  view_1066 = None
    mm_228: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1067, permute_893);  permute_893 = None
    permute_894: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1067, [1, 0])
    mm_229: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_894, view_110);  permute_894 = None
    permute_895: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1067, [0], True);  view_1067 = None
    view_1068: "f32[1024]" = torch.ops.aten.reshape.default(sum_318, [1024]);  sum_318 = None
    permute_896: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_895, [1, 0]);  permute_895 = None
    view_1069: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_228, [1, 512, 1024]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_313: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1065, view_1069);  view_1065 = view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1070: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1061, [512, 1024]);  view_1061 = None
    mm_230: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
    permute_898: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1070, [1, 0])
    mm_231: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
    permute_899: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
    view_1071: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
    permute_900: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_899, [1, 0]);  permute_899 = None
    view_1072: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_230, [1, 512, 1024]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_314: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_313, view_1072);  add_313 = view_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_744: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, primals_84);  primals_84 = None
    mul_745: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_744, 1024)
    sum_320: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True)
    mul_746: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_744, mul_36);  mul_744 = None
    sum_321: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True);  mul_746 = None
    mul_747: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, sum_321);  sum_321 = None
    sub_215: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_745, sum_320);  mul_745 = sum_320 = None
    sub_216: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_215, mul_747);  sub_215 = mul_747 = None
    mul_748: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_108, sub_216);  div_108 = sub_216 = None
    mul_749: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, mul_36);  mul_36 = None
    sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1]);  mul_749 = None
    sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_315: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_312, mul_748);  add_312 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_58: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_750: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_751: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_315, mul_750);  mul_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1073: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_751, [512, 1024]);  mul_751 = None
    mm_232: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
    permute_902: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1073, [1, 0])
    mm_233: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
    permute_903: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_324: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
    view_1074: "f32[1024]" = torch.ops.aten.reshape.default(sum_324, [1024]);  sum_324 = None
    permute_904: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
    view_1075: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_232, [1, 512, 4096]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_753: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_754: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_755: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_754, -0.5);  mul_754 = None
    exp_46: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_755);  mul_755 = None
    mul_756: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_757: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, mul_756);  view_107 = mul_756 = None
    add_317: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_753, mul_757);  mul_753 = mul_757 = None
    mul_758: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1075, add_317);  view_1075 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1076: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_758, [512, 4096]);  mul_758 = None
    mm_234: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
    permute_906: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1076, [1, 0])
    mm_235: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
    permute_907: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_325: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
    view_1077: "f32[4096]" = torch.ops.aten.reshape.default(sum_325, [4096]);  sum_325 = None
    permute_908: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_907, [1, 0]);  permute_907 = None
    view_1078: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_234, [1, 512, 1024]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_760: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, primals_78);  primals_78 = None
    mul_761: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_760, 1024)
    sum_326: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [2], True)
    mul_762: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_760, mul_31);  mul_760 = None
    sum_327: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_762, [2], True);  mul_762 = None
    mul_763: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, sum_327);  sum_327 = None
    sub_218: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_761, sum_326);  mul_761 = sum_326 = None
    sub_219: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_218, mul_763);  sub_218 = mul_763 = None
    mul_764: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_109, sub_219);  div_109 = sub_219 = None
    mul_765: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, mul_31);  mul_31 = None
    sum_328: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_765, [0, 1]);  mul_765 = None
    sum_329: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1078, [0, 1]);  view_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_318: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_315, mul_764);  add_315 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_59: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_766: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_767: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_318, mul_766);  mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1079: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_767, [512, 1024]);  mul_767 = None
    mm_236: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
    permute_910: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1079, [1, 0])
    mm_237: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
    permute_911: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_330: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
    view_1080: "f32[1024]" = torch.ops.aten.reshape.default(sum_330, [1024]);  sum_330 = None
    permute_912: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_911, [1, 0]);  permute_911 = None
    view_1081: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_236, [1, 512, 1024]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1082: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1081, [1, 512, 16, 64]);  view_1081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_913: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None
    
    # No stacktrace found for following nodes
    view_default_234: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_913, [16, 512, 64]);  permute_913 = None
    bmm_default_116: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_115, view_default_234);  permute_default_115 = None
    view_default_235: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_116, [1, 16, 512, 64]);  bmm_default_116 = None
    bmm_default_117: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_234, permute_default_116);  view_default_234 = permute_default_116 = None
    view_default_236: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_117, [1, 16, 512, 512]);  bmm_default_117 = None
    mul_tensor_77: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_236, mul_tensor_76);  view_default_236 = mul_tensor_76 = None
    mul_tensor_78: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_77, alias_default_39);  mul_tensor_77 = None
    sum_dim_int_list_39: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_78, [-1], True)
    mul_tensor_79: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_39, sum_dim_int_list_39);  alias_default_39 = sum_dim_int_list_39 = None
    sub_tensor_39: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
    view_default_237: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_39, [16, 512, 512]);  sub_tensor_39 = None
    bmm_default_118: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_117, view_default_237);  permute_default_117 = None
    view_default_238: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_118, [1, 16, 64, 512]);  bmm_default_118 = None
    mul_scalar_78: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_238, 0.3535533905932738);  view_default_238 = None
    permute_default_119: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_78, [0, 1, 3, 2]);  mul_scalar_78 = None
    bmm_default_119: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_237, permute_default_118);  view_default_237 = permute_default_118 = None
    view_default_239: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_119, [1, 16, 512, 64]);  bmm_default_119 = None
    mul_scalar_79: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_239, 0.3535533905932738);  view_default_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_919: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_79, [0, 2, 1, 3]);  mul_scalar_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_122: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
    view_1089: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_122, [1, 512, 1024]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_920: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_235, [0, 2, 1, 3]);  view_default_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_123: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_920, memory_format = torch.contiguous_format);  permute_920 = None
    view_1090: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_123, [1, 512, 1024]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1091: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1090, [512, 1024]);  view_1090 = None
    mm_238: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1091, permute_921);  permute_921 = None
    permute_922: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1091, [1, 0])
    mm_239: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_922, view_88);  permute_922 = None
    permute_923: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_332: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1091, [0], True);  view_1091 = None
    view_1092: "f32[1024]" = torch.ops.aten.reshape.default(sum_332, [1024]);  sum_332 = None
    permute_924: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_923, [1, 0]);  permute_923 = None
    view_1093: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_238, [1, 512, 1024]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_925: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_119, [0, 2, 1, 3]);  permute_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1094: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_925, [1, 512, 1024]);  permute_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1095: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1094, [512, 1024]);  view_1094 = None
    mm_240: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1095, permute_926);  permute_926 = None
    permute_927: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1095, [1, 0])
    mm_241: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_927, view_88);  permute_927 = None
    permute_928: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1095, [0], True);  view_1095 = None
    view_1096: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
    permute_929: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_928, [1, 0]);  permute_928 = None
    view_1097: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_240, [1, 512, 1024]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_319: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1093, view_1097);  view_1093 = view_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1098: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1089, [512, 1024]);  view_1089 = None
    mm_242: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
    permute_931: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1098, [1, 0])
    mm_243: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
    permute_932: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_334: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
    view_1099: "f32[1024]" = torch.ops.aten.reshape.default(sum_334, [1024]);  sum_334 = None
    permute_933: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_932, [1, 0]);  permute_932 = None
    view_1100: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_242, [1, 512, 1024]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_320: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_319, view_1100);  add_319 = view_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_773: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, primals_68);  primals_68 = None
    mul_774: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_773, 1024)
    sum_335: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True)
    mul_775: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_773, mul_29);  mul_773 = None
    sum_336: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
    mul_776: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, sum_336);  sum_336 = None
    sub_222: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_774, sum_335);  mul_774 = sum_335 = None
    sub_223: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_222, mul_776);  sub_222 = mul_776 = None
    mul_777: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_111, sub_223);  div_111 = sub_223 = None
    mul_778: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_29);  mul_29 = None
    sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 1]);  mul_778 = None
    sum_338: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_321: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_318, mul_777);  add_318 = mul_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_61: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_779: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_780: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_321, mul_779);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1101: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_780, [512, 1024]);  mul_780 = None
    mm_244: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
    permute_935: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1101, [1, 0])
    mm_245: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
    permute_936: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_339: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
    view_1102: "f32[1024]" = torch.ops.aten.reshape.default(sum_339, [1024]);  sum_339 = None
    permute_937: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_936, [1, 0]);  permute_936 = None
    view_1103: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_244, [1, 512, 4096]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_782: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_783: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_784: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_783, -0.5);  mul_783 = None
    exp_47: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_784);  mul_784 = None
    mul_785: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_786: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, mul_785);  view_85 = mul_785 = None
    add_323: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_782, mul_786);  mul_782 = mul_786 = None
    mul_787: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1103, add_323);  view_1103 = add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1104: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_787, [512, 4096]);  mul_787 = None
    mm_246: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
    permute_939: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1104, [1, 0])
    mm_247: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
    permute_940: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_340: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
    view_1105: "f32[4096]" = torch.ops.aten.reshape.default(sum_340, [4096]);  sum_340 = None
    permute_941: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_940, [1, 0]);  permute_940 = None
    view_1106: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_246, [1, 512, 1024]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_789: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, primals_62);  primals_62 = None
    mul_790: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_789, 1024)
    sum_341: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
    mul_791: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_789, mul_24);  mul_789 = None
    sum_342: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
    mul_792: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, sum_342);  sum_342 = None
    sub_225: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_790, sum_341);  mul_790 = sum_341 = None
    sub_226: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_225, mul_792);  sub_225 = mul_792 = None
    mul_793: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_112, sub_226);  div_112 = sub_226 = None
    mul_794: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, mul_24);  mul_24 = None
    sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
    sum_344: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1106, [0, 1]);  view_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_324: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_321, mul_793);  add_321 = mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_62: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_795: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_796: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_324, mul_795);  mul_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1107: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_796, [512, 1024]);  mul_796 = None
    mm_248: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
    permute_943: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1107, [1, 0])
    mm_249: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
    permute_944: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_345: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
    view_1108: "f32[1024]" = torch.ops.aten.reshape.default(sum_345, [1024]);  sum_345 = None
    permute_945: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_944, [1, 0]);  permute_944 = None
    view_1109: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_248, [1, 512, 1024]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1110: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1109, [1, 512, 16, 64]);  view_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_946: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None
    
    # No stacktrace found for following nodes
    view_default_246: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_946, [16, 512, 64]);  permute_946 = None
    bmm_default_122: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_121, view_default_246);  permute_default_121 = None
    view_default_247: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_122, [1, 16, 512, 64]);  bmm_default_122 = None
    bmm_default_123: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_246, permute_default_122);  view_default_246 = permute_default_122 = None
    view_default_248: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_123, [1, 16, 512, 512]);  bmm_default_123 = None
    mul_tensor_81: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_248, mul_tensor_80);  view_default_248 = mul_tensor_80 = None
    mul_tensor_82: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_81, alias_default_41);  mul_tensor_81 = None
    sum_dim_int_list_41: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_82, [-1], True)
    mul_tensor_83: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_41, sum_dim_int_list_41);  alias_default_41 = sum_dim_int_list_41 = None
    sub_tensor_41: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_82, mul_tensor_83);  mul_tensor_82 = mul_tensor_83 = None
    view_default_249: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_41, [16, 512, 512]);  sub_tensor_41 = None
    bmm_default_124: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_123, view_default_249);  permute_default_123 = None
    view_default_250: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_124, [1, 16, 64, 512]);  bmm_default_124 = None
    mul_scalar_82: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_250, 0.3535533905932738);  view_default_250 = None
    permute_default_125: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_82, [0, 1, 3, 2]);  mul_scalar_82 = None
    bmm_default_125: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_249, permute_default_124);  view_default_249 = permute_default_124 = None
    view_default_251: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_125, [1, 16, 512, 64]);  bmm_default_125 = None
    mul_scalar_83: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_251, 0.3535533905932738);  view_default_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_952: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_83, [0, 2, 1, 3]);  mul_scalar_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_127: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
    view_1117: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_127, [1, 512, 1024]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_953: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_247, [0, 2, 1, 3]);  view_default_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_128: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_953, memory_format = torch.contiguous_format);  permute_953 = None
    view_1118: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_128, [1, 512, 1024]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1119: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1118, [512, 1024]);  view_1118 = None
    mm_250: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1119, permute_954);  permute_954 = None
    permute_955: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1119, [1, 0])
    mm_251: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_955, view_66);  permute_955 = None
    permute_956: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1119, [0], True);  view_1119 = None
    view_1120: "f32[1024]" = torch.ops.aten.reshape.default(sum_347, [1024]);  sum_347 = None
    permute_957: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_956, [1, 0]);  permute_956 = None
    view_1121: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_250, [1, 512, 1024]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_958: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_125, [0, 2, 1, 3]);  permute_default_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1122: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_958, [1, 512, 1024]);  permute_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1123: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1122, [512, 1024]);  view_1122 = None
    mm_252: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1123, permute_959);  permute_959 = None
    permute_960: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1123, [1, 0])
    mm_253: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_960, view_66);  permute_960 = None
    permute_961: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_348: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1123, [0], True);  view_1123 = None
    view_1124: "f32[1024]" = torch.ops.aten.reshape.default(sum_348, [1024]);  sum_348 = None
    permute_962: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_961, [1, 0]);  permute_961 = None
    view_1125: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_252, [1, 512, 1024]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_325: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1121, view_1125);  view_1121 = view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1126: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1117, [512, 1024]);  view_1117 = None
    mm_254: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
    permute_964: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1126, [1, 0])
    mm_255: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
    permute_965: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_349: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
    view_1127: "f32[1024]" = torch.ops.aten.reshape.default(sum_349, [1024]);  sum_349 = None
    permute_966: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_965, [1, 0]);  permute_965 = None
    view_1128: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_254, [1, 512, 1024]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_326: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_325, view_1128);  add_325 = view_1128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_802: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, primals_52);  primals_52 = None
    mul_803: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_802, 1024)
    sum_350: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_802, [2], True)
    mul_804: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_802, mul_22);  mul_802 = None
    sum_351: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True);  mul_804 = None
    mul_805: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, sum_351);  sum_351 = None
    sub_229: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_803, sum_350);  mul_803 = sum_350 = None
    sub_230: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_229, mul_805);  sub_229 = mul_805 = None
    mul_806: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_114, sub_230);  div_114 = sub_230 = None
    mul_807: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, mul_22);  mul_22 = None
    sum_352: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 1]);  mul_807 = None
    sum_353: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_327: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_324, mul_806);  add_324 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_808: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_809: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_327, mul_808);  mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1129: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_809, [512, 1024]);  mul_809 = None
    mm_256: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
    permute_968: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1129, [1, 0])
    mm_257: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
    permute_969: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_354: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
    view_1130: "f32[1024]" = torch.ops.aten.reshape.default(sum_354, [1024]);  sum_354 = None
    permute_970: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_969, [1, 0]);  permute_969 = None
    view_1131: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_256, [1, 512, 4096]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_811: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_812: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_813: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_48: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_815: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, mul_814);  view_63 = mul_814 = None
    add_329: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1131, add_329);  view_1131 = add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1132: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_816, [512, 4096]);  mul_816 = None
    mm_258: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
    permute_972: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1132, [1, 0])
    mm_259: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
    permute_973: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_355: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
    view_1133: "f32[4096]" = torch.ops.aten.reshape.default(sum_355, [4096]);  sum_355 = None
    permute_974: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_973, [1, 0]);  permute_973 = None
    view_1134: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_258, [1, 512, 1024]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_818: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, primals_46);  primals_46 = None
    mul_819: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_818, 1024)
    sum_356: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [2], True)
    mul_820: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_818, mul_17);  mul_818 = None
    sum_357: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
    mul_821: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, sum_357);  sum_357 = None
    sub_232: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_819, sum_356);  mul_819 = sum_356 = None
    sub_233: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_232, mul_821);  sub_232 = mul_821 = None
    mul_822: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_115, sub_233);  div_115 = sub_233 = None
    mul_823: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, mul_17);  mul_17 = None
    sum_358: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
    sum_359: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1134, [0, 1]);  view_1134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_330: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_327, mul_822);  add_327 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_65: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_824: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_825: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_330, mul_824);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1135: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_825, [512, 1024]);  mul_825 = None
    mm_260: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
    permute_976: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1135, [1, 0])
    mm_261: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
    permute_977: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_360: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
    view_1136: "f32[1024]" = torch.ops.aten.reshape.default(sum_360, [1024]);  sum_360 = None
    permute_978: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_977, [1, 0]);  permute_977 = None
    view_1137: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_260, [1, 512, 1024]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1138: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1137, [1, 512, 16, 64]);  view_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_979: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None
    
    # No stacktrace found for following nodes
    view_default_258: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_979, [16, 512, 64]);  permute_979 = None
    bmm_default_128: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_127, view_default_258);  permute_default_127 = None
    view_default_259: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_128, [1, 16, 512, 64]);  bmm_default_128 = None
    bmm_default_129: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_258, permute_default_128);  view_default_258 = permute_default_128 = None
    view_default_260: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_129, [1, 16, 512, 512]);  bmm_default_129 = None
    mul_tensor_85: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_260, mul_tensor_84);  view_default_260 = mul_tensor_84 = None
    mul_tensor_86: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_85, alias_default_43);  mul_tensor_85 = None
    sum_dim_int_list_43: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_86, [-1], True)
    mul_tensor_87: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_43, sum_dim_int_list_43);  alias_default_43 = sum_dim_int_list_43 = None
    sub_tensor_43: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_86, mul_tensor_87);  mul_tensor_86 = mul_tensor_87 = None
    view_default_261: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_43, [16, 512, 512]);  sub_tensor_43 = None
    bmm_default_130: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_129, view_default_261);  permute_default_129 = None
    view_default_262: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_130, [1, 16, 64, 512]);  bmm_default_130 = None
    mul_scalar_86: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_262, 0.3535533905932738);  view_default_262 = None
    permute_default_131: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_86, [0, 1, 3, 2]);  mul_scalar_86 = None
    bmm_default_131: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_261, permute_default_130);  view_default_261 = permute_default_130 = None
    view_default_263: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_131, [1, 16, 512, 64]);  bmm_default_131 = None
    mul_scalar_87: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_263, 0.3535533905932738);  view_default_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_985: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_87, [0, 2, 1, 3]);  mul_scalar_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_132: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
    view_1145: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_132, [1, 512, 1024]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_986: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_259, [0, 2, 1, 3]);  view_default_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_133: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_986, memory_format = torch.contiguous_format);  permute_986 = None
    view_1146: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_133, [1, 512, 1024]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1147: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1146, [512, 1024]);  view_1146 = None
    mm_262: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1147, permute_987);  permute_987 = None
    permute_988: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1147, [1, 0])
    mm_263: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_988, view_44);  permute_988 = None
    permute_989: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1147, [0], True);  view_1147 = None
    view_1148: "f32[1024]" = torch.ops.aten.reshape.default(sum_362, [1024]);  sum_362 = None
    permute_990: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_989, [1, 0]);  permute_989 = None
    view_1149: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_262, [1, 512, 1024]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_991: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_131, [0, 2, 1, 3]);  permute_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1150: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_991, [1, 512, 1024]);  permute_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1151: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1150, [512, 1024]);  view_1150 = None
    mm_264: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1151, permute_992);  permute_992 = None
    permute_993: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1151, [1, 0])
    mm_265: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_993, view_44);  permute_993 = None
    permute_994: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_363: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True);  view_1151 = None
    view_1152: "f32[1024]" = torch.ops.aten.reshape.default(sum_363, [1024]);  sum_363 = None
    permute_995: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_994, [1, 0]);  permute_994 = None
    view_1153: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_264, [1, 512, 1024]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_331: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1149, view_1153);  view_1149 = view_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1154: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1145, [512, 1024]);  view_1145 = None
    mm_266: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
    permute_997: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1154, [1, 0])
    mm_267: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
    permute_998: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_364: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
    view_1155: "f32[1024]" = torch.ops.aten.reshape.default(sum_364, [1024]);  sum_364 = None
    permute_999: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_998, [1, 0]);  permute_998 = None
    view_1156: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_266, [1, 512, 1024]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_332: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_331, view_1156);  add_331 = view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_831: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, primals_36);  primals_36 = None
    mul_832: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, 1024)
    sum_365: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
    mul_833: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, mul_15);  mul_831 = None
    sum_366: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
    mul_834: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, sum_366);  sum_366 = None
    sub_236: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_832, sum_365);  mul_832 = sum_365 = None
    sub_237: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_236, mul_834);  sub_236 = mul_834 = None
    mul_835: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_117, sub_237);  div_117 = sub_237 = None
    mul_836: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_15);  mul_15 = None
    sum_367: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
    sum_368: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_333: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_330, mul_835);  add_330 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_67: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_837: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_838: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_333, mul_837);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1157: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_838, [512, 1024]);  mul_838 = None
    mm_268: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
    permute_1001: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1157, [1, 0])
    mm_269: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
    permute_1002: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_369: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
    view_1158: "f32[1024]" = torch.ops.aten.reshape.default(sum_369, [1024]);  sum_369 = None
    permute_1003: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1002, [1, 0]);  permute_1002 = None
    view_1159: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_268, [1, 512, 4096]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_840: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_841: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_842: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_841, -0.5);  mul_841 = None
    exp_49: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_842);  mul_842 = None
    mul_843: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_844: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, mul_843);  view_41 = mul_843 = None
    add_335: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_840, mul_844);  mul_840 = mul_844 = None
    mul_845: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1159, add_335);  view_1159 = add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1160: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_845, [512, 4096]);  mul_845 = None
    mm_270: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
    permute_1005: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1160, [1, 0])
    mm_271: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
    permute_1006: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_370: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
    view_1161: "f32[4096]" = torch.ops.aten.reshape.default(sum_370, [4096]);  sum_370 = None
    permute_1007: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1006, [1, 0]);  permute_1006 = None
    view_1162: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_270, [1, 512, 1024]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_847: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, primals_30);  primals_30 = None
    mul_848: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_847, 1024)
    sum_371: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_847, [2], True)
    mul_849: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_847, mul_10);  mul_847 = None
    sum_372: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_849, [2], True);  mul_849 = None
    mul_850: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, sum_372);  sum_372 = None
    sub_239: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_848, sum_371);  mul_848 = sum_371 = None
    sub_240: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_239, mul_850);  sub_239 = mul_850 = None
    mul_851: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_118, sub_240);  div_118 = sub_240 = None
    mul_852: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, mul_10);  mul_10 = None
    sum_373: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 1]);  mul_852 = None
    sum_374: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1162, [0, 1]);  view_1162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_336: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_333, mul_851);  add_333 = mul_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_853: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_854: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_336, mul_853);  mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1163: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_854, [512, 1024]);  mul_854 = None
    mm_272: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
    permute_1009: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1163, [1, 0])
    mm_273: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
    permute_1010: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_375: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
    view_1164: "f32[1024]" = torch.ops.aten.reshape.default(sum_375, [1024]);  sum_375 = None
    permute_1011: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1010, [1, 0]);  permute_1010 = None
    view_1165: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_272, [1, 512, 1024]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1166: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1165, [1, 512, 16, 64]);  view_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1012: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None
    
    # No stacktrace found for following nodes
    view_default_270: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1012, [16, 512, 64]);  permute_1012 = None
    bmm_default_134: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_133, view_default_270);  permute_default_133 = None
    view_default_271: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_134, [1, 16, 512, 64]);  bmm_default_134 = None
    bmm_default_135: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_270, permute_default_134);  view_default_270 = permute_default_134 = None
    view_default_272: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_135, [1, 16, 512, 512]);  bmm_default_135 = None
    mul_tensor_89: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_272, mul_tensor_88);  view_default_272 = mul_tensor_88 = None
    mul_tensor_90: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_89, alias_default_45);  mul_tensor_89 = None
    sum_dim_int_list_45: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_90, [-1], True)
    mul_tensor_91: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_45, sum_dim_int_list_45);  alias_default_45 = sum_dim_int_list_45 = None
    sub_tensor_45: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
    view_default_273: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_45, [16, 512, 512]);  sub_tensor_45 = None
    bmm_default_136: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_135, view_default_273);  permute_default_135 = None
    view_default_274: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_136, [1, 16, 64, 512]);  bmm_default_136 = None
    mul_scalar_90: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_274, 0.3535533905932738);  view_default_274 = None
    permute_default_137: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_90, [0, 1, 3, 2]);  mul_scalar_90 = None
    bmm_default_137: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_273, permute_default_136);  view_default_273 = permute_default_136 = None
    view_default_275: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_137, [1, 16, 512, 64]);  bmm_default_137 = None
    mul_scalar_91: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_275, 0.3535533905932738);  view_default_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1018: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_91, [0, 2, 1, 3]);  mul_scalar_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_137: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1018, memory_format = torch.contiguous_format);  permute_1018 = None
    view_1173: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_137, [1, 512, 1024]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1019: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_271, [0, 2, 1, 3]);  view_default_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_138: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1019, memory_format = torch.contiguous_format);  permute_1019 = None
    view_1174: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_138, [1, 512, 1024]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1175: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1174, [512, 1024]);  view_1174 = None
    mm_274: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1175, permute_1020);  permute_1020 = None
    permute_1021: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1175, [1, 0])
    mm_275: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1021, view_22);  permute_1021 = None
    permute_1022: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1175, [0], True);  view_1175 = None
    view_1176: "f32[1024]" = torch.ops.aten.reshape.default(sum_377, [1024]);  sum_377 = None
    permute_1023: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1022, [1, 0]);  permute_1022 = None
    view_1177: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_274, [1, 512, 1024]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1024: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_137, [0, 2, 1, 3]);  permute_default_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1178: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1024, [1, 512, 1024]);  permute_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1179: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1178, [512, 1024]);  view_1178 = None
    mm_276: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1179, permute_1025);  permute_1025 = None
    permute_1026: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1179, [1, 0])
    mm_277: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1026, view_22);  permute_1026 = None
    permute_1027: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_378: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1179, [0], True);  view_1179 = None
    view_1180: "f32[1024]" = torch.ops.aten.reshape.default(sum_378, [1024]);  sum_378 = None
    permute_1028: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1027, [1, 0]);  permute_1027 = None
    view_1181: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_276, [1, 512, 1024]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_337: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1177, view_1181);  view_1177 = view_1181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1182: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1173, [512, 1024]);  view_1173 = None
    mm_278: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
    permute_1030: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1182, [1, 0])
    mm_279: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
    permute_1031: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_379: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
    view_1183: "f32[1024]" = torch.ops.aten.reshape.default(sum_379, [1024]);  sum_379 = None
    permute_1032: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1031, [1, 0]);  permute_1031 = None
    view_1184: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_278, [1, 512, 1024]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_338: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_337, view_1184);  add_337 = view_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_860: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, primals_20);  primals_20 = None
    mul_861: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_860, 1024)
    sum_380: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_860, [2], True)
    mul_862: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_860, mul_8);  mul_860 = None
    sum_381: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_862, [2], True);  mul_862 = None
    mul_863: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_8, sum_381);  sum_381 = None
    sub_243: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_861, sum_380);  mul_861 = sum_380 = None
    sub_244: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_243, mul_863);  sub_243 = mul_863 = None
    mul_864: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_120, sub_244);  div_120 = sub_244 = None
    mul_865: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_8);  mul_8 = None
    sum_382: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 1]);  mul_865 = None
    sum_383: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_339: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_336, mul_864);  add_336 = mul_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_70: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_866: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_867: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_339, mul_866);  mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1185: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_867, [512, 1024]);  mul_867 = None
    mm_280: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
    permute_1034: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1185, [1, 0])
    mm_281: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
    permute_1035: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_384: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
    view_1186: "f32[1024]" = torch.ops.aten.reshape.default(sum_384, [1024]);  sum_384 = None
    permute_1036: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1035, [1, 0]);  permute_1035 = None
    view_1187: "f32[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_280, [1, 512, 4096]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_869: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_870: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_871: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_870, -0.5);  mul_870 = None
    exp_50: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_871);  mul_871 = None
    mul_872: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_873: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, mul_872);  view_19 = mul_872 = None
    add_341: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_869, mul_873);  mul_869 = mul_873 = None
    mul_874: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1187, add_341);  view_1187 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1188: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_874, [512, 4096]);  mul_874 = None
    mm_282: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
    permute_1038: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1188, [1, 0])
    mm_283: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
    permute_1039: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_385: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
    view_1189: "f32[4096]" = torch.ops.aten.reshape.default(sum_385, [4096]);  sum_385 = None
    permute_1040: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1039, [1, 0]);  permute_1039 = None
    view_1190: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_282, [1, 512, 1024]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_876: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, primals_14);  primals_14 = None
    mul_877: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_876, 1024)
    sum_386: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2], True)
    mul_878: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_876, mul_3);  mul_876 = None
    sum_387: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_878, [2], True);  mul_878 = None
    mul_879: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_387);  sum_387 = None
    sub_246: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_877, sum_386);  mul_877 = sum_386 = None
    sub_247: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_246, mul_879);  sub_246 = mul_879 = None
    mul_880: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_121, sub_247);  div_121 = sub_247 = None
    mul_881: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, mul_3);  mul_3 = None
    sum_388: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_881, [0, 1]);  mul_881 = None
    sum_389: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1190, [0, 1]);  view_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_342: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_339, mul_880);  add_339 = mul_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_71: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_882: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_883: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_342, mul_882);  mul_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1191: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_883, [512, 1024]);  mul_883 = None
    mm_284: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
    permute_1042: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1191, [1, 0])
    mm_285: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
    permute_1043: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_390: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
    view_1192: "f32[1024]" = torch.ops.aten.reshape.default(sum_390, [1024]);  sum_390 = None
    permute_1044: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1043, [1, 0]);  permute_1043 = None
    view_1193: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_284, [1, 512, 1024]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1194: "f32[1, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1193, [1, 512, 16, 64]);  view_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1045: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None
    
    # No stacktrace found for following nodes
    view_default_282: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1045, [16, 512, 64]);  permute_1045 = None
    bmm_default_140: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_139, view_default_282);  permute_default_139 = None
    view_default_283: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_140, [1, 16, 512, 64]);  bmm_default_140 = None
    bmm_default_141: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_282, permute_default_140);  view_default_282 = permute_default_140 = None
    view_default_284: "f32[1, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_141, [1, 16, 512, 512]);  bmm_default_141 = None
    mul_tensor_93: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_284, mul_tensor_92);  view_default_284 = mul_tensor_92 = None
    mul_tensor_94: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_93, alias_default_47);  mul_tensor_93 = None
    sum_dim_int_list_47: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_94, [-1], True)
    mul_tensor_95: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_47, sum_dim_int_list_47);  alias_default_47 = sum_dim_int_list_47 = None
    sub_tensor_47: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_94, mul_tensor_95);  mul_tensor_94 = mul_tensor_95 = None
    view_default_285: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_47, [16, 512, 512]);  sub_tensor_47 = None
    bmm_default_142: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_141, view_default_285);  permute_default_141 = None
    view_default_286: "f32[1, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_142, [1, 16, 64, 512]);  bmm_default_142 = None
    mul_scalar_94: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_286, 0.3535533905932738);  view_default_286 = None
    permute_default_143: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_94, [0, 1, 3, 2]);  mul_scalar_94 = None
    bmm_default_143: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_285, permute_default_142);  view_default_285 = permute_default_142 = None
    view_default_287: "f32[1, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_143, [1, 16, 512, 64]);  bmm_default_143 = None
    mul_scalar_95: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_287, 0.3535533905932738);  view_default_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1051: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_95, [0, 2, 1, 3]);  mul_scalar_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_142: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1051, memory_format = torch.contiguous_format);  permute_1051 = None
    view_1201: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_142, [1, 512, 1024]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1052: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_283, [0, 2, 1, 3]);  view_default_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_143: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1052, memory_format = torch.contiguous_format);  permute_1052 = None
    view_1202: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_143, [1, 512, 1024]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1203: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1202, [512, 1024]);  view_1202 = None
    mm_286: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1203, permute_1053);  permute_1053 = None
    permute_1054: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1203, [1, 0])
    mm_287: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1054, view);  permute_1054 = None
    permute_1055: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1203, [0], True);  view_1203 = None
    view_1204: "f32[1024]" = torch.ops.aten.reshape.default(sum_392, [1024]);  sum_392 = None
    permute_1056: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1055, [1, 0]);  permute_1055 = None
    view_1205: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, [1, 512, 1024]);  mm_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1057: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_143, [0, 2, 1, 3]);  permute_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1206: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1057, [1, 512, 1024]);  permute_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1207: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1206, [512, 1024]);  view_1206 = None
    mm_288: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1207, permute_1058);  permute_1058 = None
    permute_1059: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1207, [1, 0])
    mm_289: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1059, view);  permute_1059 = None
    permute_1060: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    sum_393: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True);  view_1207 = None
    view_1208: "f32[1024]" = torch.ops.aten.reshape.default(sum_393, [1024]);  sum_393 = None
    permute_1061: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1060, [1, 0]);  permute_1060 = None
    view_1209: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, [1, 512, 1024]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_343: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1205, view_1209);  view_1205 = view_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1210: "f32[512, 1024]" = torch.ops.aten.reshape.default(view_1201, [512, 1024]);  view_1201 = None
    mm_290: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
    permute_1063: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1210, [1, 0])
    mm_291: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
    permute_1064: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_291, [1, 0]);  mm_291 = None
    sum_394: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
    view_1211: "f32[1024]" = torch.ops.aten.reshape.default(sum_394, [1024]);  sum_394 = None
    permute_1065: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1064, [1, 0]);  permute_1064 = None
    view_1212: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, [1, 512, 1024]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_344: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_343, view_1212);  add_343 = view_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_889: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, primals_4);  primals_4 = None
    mul_890: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_889, 1024)
    sum_395: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True)
    mul_891: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_889, mul_1);  mul_889 = None
    sum_396: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_891, [2], True);  mul_891 = None
    mul_892: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, sum_396);  sum_396 = None
    sub_250: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_890, sum_395);  mul_890 = sum_395 = None
    sub_251: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_250, mul_892);  sub_250 = mul_892 = None
    mul_893: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_251);  div_123 = sub_251 = None
    mul_894: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, mul_1);  mul_1 = None
    sum_397: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 1]);  mul_894 = None
    sum_398: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_345: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_342, mul_893);  add_342 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189, code: embeddings = self.dropout(embeddings)
    convert_element_type_73: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_895: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
    mul_896: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_345, mul_895);  add_345 = mul_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:184, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_3, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_4, full_default_3, mul_896);  unsqueeze_4 = None
    full_default_11: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 1024]" = torch.ops.prims._unsafe_index_put_.default(full_default_11, [slice_3], where_4, True);  full_default_11 = slice_3 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    full_default_12: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_5: "f32[1, 512, 1024]" = torch.ops.aten.where.self(full_default_12, full_default_3, mul_896);  full_default_12 = None
    full_default_14: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 1024]" = torch.ops.prims._unsafe_index_put_.default(full_default_14, [full_default], where_5, True);  full_default_14 = full_default = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_398, 0)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_6: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_6, full_default_3, mul_896);  unsqueeze_6 = full_default_3 = mul_896 = None
    full_default_16: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[29056, 1024]" = torch.ops.prims._unsafe_index_put_.default(full_default_16, [primals_398], where_6, True);  full_default_16 = primals_398 = where_6 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_397, sum_398, permute_1065, view_1211, permute_1061, view_1208, permute_1056, view_1204, permute_1044, view_1192, sum_388, sum_389, permute_1040, view_1189, permute_1036, view_1186, sum_382, sum_383, permute_1032, view_1183, permute_1028, view_1180, permute_1023, view_1176, permute_1011, view_1164, sum_373, sum_374, permute_1007, view_1161, permute_1003, view_1158, sum_367, sum_368, permute_999, view_1155, permute_995, view_1152, permute_990, view_1148, permute_978, view_1136, sum_358, sum_359, permute_974, view_1133, permute_970, view_1130, sum_352, sum_353, permute_966, view_1127, permute_962, view_1124, permute_957, view_1120, permute_945, view_1108, sum_343, sum_344, permute_941, view_1105, permute_937, view_1102, sum_337, sum_338, permute_933, view_1099, permute_929, view_1096, permute_924, view_1092, permute_912, view_1080, sum_328, sum_329, permute_908, view_1077, permute_904, view_1074, sum_322, sum_323, permute_900, view_1071, permute_896, view_1068, permute_891, view_1064, permute_879, view_1052, sum_313, sum_314, permute_875, view_1049, permute_871, view_1046, sum_307, sum_308, permute_867, view_1043, permute_863, view_1040, permute_858, view_1036, permute_846, view_1024, sum_298, sum_299, permute_842, view_1021, permute_838, view_1018, sum_292, sum_293, permute_834, view_1015, permute_830, view_1012, permute_825, view_1008, permute_813, view_996, sum_283, sum_284, permute_809, view_993, permute_805, view_990, sum_277, sum_278, permute_801, view_987, permute_797, view_984, permute_792, view_980, permute_780, view_968, sum_268, sum_269, permute_776, view_965, permute_772, view_962, sum_262, sum_263, permute_768, view_959, permute_764, view_956, permute_759, view_952, permute_747, view_940, sum_253, sum_254, permute_743, view_937, permute_739, view_934, sum_247, sum_248, permute_735, view_931, permute_731, view_928, permute_726, view_924, permute_714, view_912, sum_238, sum_239, permute_710, view_909, permute_706, view_906, sum_232, sum_233, permute_702, view_903, permute_698, view_900, permute_693, view_896, permute_681, view_884, sum_223, sum_224, permute_677, view_881, permute_673, view_878, sum_217, sum_218, permute_669, view_875, permute_665, view_872, permute_660, view_868, permute_648, view_856, sum_208, sum_209, permute_644, view_853, permute_640, view_850, sum_202, sum_203, permute_636, view_847, permute_632, view_844, permute_627, view_840, permute_615, view_828, sum_193, sum_194, permute_611, view_825, permute_607, view_822, sum_187, sum_188, permute_603, view_819, permute_599, view_816, permute_594, view_812, permute_582, view_800, sum_178, sum_179, permute_578, view_797, permute_574, view_794, sum_172, sum_173, permute_570, view_791, permute_566, view_788, permute_561, view_784, permute_549, view_772, sum_163, sum_164, permute_545, view_769, permute_541, view_766, sum_157, sum_158, permute_537, view_763, permute_533, view_760, permute_528, view_756, permute_516, view_744, sum_148, sum_149, permute_512, view_741, permute_508, view_738, sum_142, sum_143, permute_504, view_735, permute_500, view_732, permute_495, view_728, permute_483, view_716, sum_133, sum_134, permute_479, view_713, permute_475, view_710, sum_127, sum_128, permute_471, view_707, permute_467, view_704, permute_462, view_700, permute_450, view_688, sum_118, sum_119, permute_446, view_685, permute_442, view_682, sum_112, sum_113, permute_438, view_679, permute_434, view_676, permute_429, view_672, permute_417, view_660, sum_103, sum_104, permute_413, view_657, permute_409, view_654, sum_97, sum_98, permute_405, view_651, permute_401, view_648, permute_396, view_644, permute_384, view_632, sum_88, sum_89, permute_380, view_629, permute_376, view_626, sum_82, sum_83, permute_372, view_623, permute_368, view_620, permute_363, view_616, permute_351, view_604, sum_73, sum_74, permute_347, view_601, permute_343, view_598, sum_67, sum_68, permute_339, view_595, permute_335, view_592, permute_330, view_588, permute_318, view_576, sum_58, sum_59, permute_314, view_573, permute_310, view_570, sum_52, sum_53, permute_306, view_567, permute_302, view_564, permute_297, view_560, permute_285, view_548, sum_43, sum_44, permute_281, view_545, permute_277, view_542, sum_37, sum_38, permute_273, view_539, sum_32, sum_33, permute_269, view_536, None, None, None]
    